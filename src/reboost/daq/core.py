from __future__ import annotations

import awkward as ak
import numba
import numpy as np

from .utils import print_random_crash_msg


def run_daq_non_sparse(
    evt: ak.Array,
    n_sim_events: int,
    source_activity: float,
    *,
    tau_preamp: float = 500,
    noise_threshold: float = 5,
    baseline_slope_threshold: float = 0.01,
    trigger_threshold: float = 25,
    waveform_length: float = 100,
    trigger_position: float = 50,
):
    r"""Run the DAQ in non-sparse mode.

    Pipe simulated HPGe events through the DAQ system in non-sparse mode.
    Return a table where each row represents an event that was actually
    recorded by the DAQ. for each event and each channel, determine the
    characteristics of the waveform.

    Warning
    -------
    This code assumes that the simulated events are time-independent.

    The returned Awkward array (the table) has the following fields:

    - ``evtid`` (int): event ID in the simulation.
    - ``timestamp`` (float): timestamp of the event.
    - ``has_trigger`` (array of bools): this waveform triggered the DAQ.
    - ``has_pre_pulse`` (array of bools): the waveform has a signal below
      trigger_threshold in the first part of the waveform, before the trigger
      position.
    - ``has_post_pulse`` (array of bools): the waveform has a signal in the
      second part of the waveform, after the trigger position.
    - ``has_slope`` (array of bools): waveform has decaying tail of earlier
      signals, that came before this waveform.

    The table sits in a tuple, together with a list of the channel
    identifiers, with the same order as in the data array.

    Parameters
    ----------
    evt
        simulated events.
    source_activity
        source activity in Bq.
    n_sim_events
        total number of simulated events.
    tau_preamp
        pre-amplification RC constant in microseconds. the signal model is an
        exponential:

        .. math::

          f(t) = E_i * e^{((t - t_i) / \tau)}

        where :math:`E_i` is the energy of the signal and :math:`t_i` is the
        time it occurred.
    noise_threshold
        threshold (in keV) for a signal to be "visible" above noise. In
        LEGEND-200, the "energy" of forced trigger events is
        gauss-distributed around 0.5 keV with a standard deviation of about
        0.5 keV.
    baseline_slope_threshold
        threshold (in keV/us) on the baseline slope to be tagged as not flat.
        in LEGEND-200, the slope of waveforms in force-triggered events is
        gauss-distributed around 0 with a standard deviation of about 2 keV/ms.
    trigger_threshold
        amplitude (in keV) needed for the DAQ to trigger on a signal.
    waveform_length
        length of the waveform in microseconds stored on disk.
    trigger_position
        location (offset) in microseconds of the triggered signal in the
        waveform.
    """
    # random engine
    rng = np.random.default_rng()

    # simulate ORCA
    print_random_crash_msg(rng)

    # add time of each simulated event, in microseconds, according to the expected event rate
    detected_rate = source_activity * len(evt) / n_sim_events
    evt["t0"] = np.cumsum(rng.exponential(scale=1e6 / detected_rate, size=len(evt)))

    # get rawids of detectors present in the simulation
    channel_ids = np.sort(np.unique(ak.flatten(evt.geds_rawid_active))).to_list()

    daq_records = _run_daq_non_sparse_impl(
        evt,
        channel_ids,
        tau_preamp,
        noise_threshold,
        baseline_slope_threshold,
        trigger_threshold,
        waveform_length,
        trigger_position,
    )

    fields = ["evtid", "timestamp", "has_trigger", "has_pre_pulse", "has_post_pulse", "has_slope"]

    daq_data = ak.Array(dict(zip(fields, daq_records, strict=False)))

    return daq_data, channel_ids


@numba.njit(cache=True)
def _run_daq_non_sparse_impl(
    evt: ak.Array,
    chids: list,
    tau_preamp: float,
    noise_threshold: float,
    baseline_slope_threshold: float,
    trigger_threshold: float,
    waveform_length: float,
    trigger_position: float,
):
    """Numba-accelerated implementation of :func:`run_daq_non_sparse`."""
    o_evtid = np.full(len(evt), dtype=np.int64, fill_value=-1)
    o_timestamp = np.full(len(evt), dtype=np.float64, fill_value=-1)

    def _init_data(dtype=np.bool_):
        return np.zeros((len(evt), len(chids)), dtype=dtype)

    o_has_trigger = _init_data()
    o_has_pre_pulse = _init_data()
    o_has_post_pulse = _init_data()
    o_has_slope = _init_data()

    # this is the index of the current daq record
    r_idx = -1

    # list of event indices (fifo) to keep track of past events that have still
    # an effect on the baseline
    evt_idx_buffer = []

    # TODO: this will need to be updated
    # {
    #  evtid: 5.37e+03,
    #  geds_energy_active: [756, 152],  in keV, "geds" means "HPGe detectors"
    #  geds_multiplicity_active: 2,
    #  geds_rawid_active: [1110400, 1112000],
    #  geds_t0_active: [1.53e+12, 1.53e+12]
    # }

    # loop over simulated events
    for s_idx, ev in enumerate(evt):
        # loop over the event_buffer and remove events that occurred more
        # than 10 times the tau_preamp ago
        cutoff = ev.t0 - 10 * tau_preamp
        while evt_idx_buffer:
            first_idx = evt_idx_buffer[0]
            if evt[first_idx].t0 < cutoff:
                evt_idx_buffer.pop(0)
            else:
                break

        # add current event to the buffer for later baseline analysis
        evt_idx_buffer.append(s_idx)

        # don't do any of this if there is no last record yet
        if r_idx != -1:
            # check if the last trigger was less than (waveform_length - trigger_position)
            # ago. if yes, this is not a trigger but there is a hard post-pile-up
            # on the previous trigger. so we need to check for each channel if the
            # energy deposited is above the noise_threshold. if yes, get the last
            # trigger and set the hard_post_pileup flag of that channel to true.
            # then continue to the next event.
            dt = ev.t0 - o_timestamp[r_idx]
            if dt < (waveform_length - trigger_position):
                for rawid, ene in zip(ev.geds_rawid_active, ev.geds_energy_active):  # noqa: B905
                    if ene >= noise_threshold:
                        o_has_post_pulse[r_idx, chids.index(rawid)] = True
                continue

            # check if the last trigger was less than waveform_length but more than
            # (waveform_length - trigger_position) ago. if yes, this event is not
            # recorded by the daq (dead time), so do nothing and just continue to the
            # next event.
            if dt > (waveform_length - trigger_position) and dt < waveform_length:
                continue

        # if we are here it means that we can actively look for new triggers.
        # check if we have a trigger by checking energy in each detector against
        # the trigger_threshold. if not, continue to the next event
        triggered_rawids = []
        for rawid, ene in zip(ev.geds_rawid_active, ev.geds_energy_active):  # noqa: B905
            if ene >= trigger_threshold:
                triggered_rawids.append(int(rawid))

        if not triggered_rawids:
            continue

        # if we are here, it means we found a trigger. first, we save the last
        # daq record to disk and we initialize a new one
        r_idx += 1

        o_evtid[r_idx] = ev.evtid
        o_timestamp[r_idx] = ev.t0

        # then, let's log which channels triggered the daq in the daq_record
        for rawid in triggered_rawids:
            o_has_trigger[r_idx, chids.index(rawid)] = True

        # time of the start of the waveform
        t0_start = ev.t0 - trigger_position
        # now we need to peek into the event_buffer to check if the baseline is
        # affected by the tails of previous events (soft pile-up) or includes a
        # small in-trace signal (pre-hard-pileup). to take a decision, we use the
        # baseline slope threshold
        for rawid in chids:
            abs_baseline_slope = 0
            for j in evt_idx_buffer:
                _ev = evt[j]

                # for each event in the buffer get timestamp and energy
                tj = _ev.t0
                ej = 0
                for k, e in zip(_ev.geds_rawid_active, _ev.geds_energy_active):  # noqa: B905
                    if k == rawid:
                        ej = e
                        break

                # if the event occurred before the current waveform window, we
                # account for its tail in the baseline of the current waveform
                # the baseline slope is calculated at the start of the waveform
                if tj <= t0_start:
                    abs_baseline_slope += ej / tau_preamp * np.exp(-(t0_start - tj) / tau_preamp)

                # if there was any energy in a channel that occurred less than
                # (timestamp - trigger_position) ago, this channel has a hard
                # pre-pile-up in the current waveform
                elif tj < ev.t0 and ej >= noise_threshold:
                    o_has_pre_pulse[r_idx, chids.index(rawid)] = True

            # now we have computed the baseline and we can check against the the
            # noise threshold if it's significantly non-flat
            if abs_baseline_slope >= baseline_slope_threshold:
                o_has_slope[r_idx, chids.index(rawid)] = True

    # the timestamp should refer to the start of the waveform, like in our DAQ
    o_timestamp -= trigger_position

    # the last event was recorded too
    r_idx += 1

    return (
        o_evtid[:r_idx],
        o_timestamp[:r_idx],
        o_has_trigger[:r_idx],
        o_has_pre_pulse[:r_idx],
        o_has_post_pulse[:r_idx],
        o_has_slope[:r_idx],
    )
