# courtesy of D. Hervas

using SolidStateDetectors # >= 5f3a5cb
using LegendTestData
using LegendDataManagement # >= ad9555c
using Plots
using Unitful
using ProgressMeter
using LegendHDF5IO
using HDF5
using Base.Threads

SSD = SolidStateDetectors
T = Float32

for (id,det) in enumerate(["V99000","B99000"])
    # let's use V10437B. Note: private metadata
    root_path = legend_test_data_path() * "/data/legend/metadata/hardware/detectors/germanium"
    meta = LegendDataManagement.readlprops("$root_path/diodes/$(det)A.yaml")
    xtal_meta = LegendDataManagement.readlprops("$root_path/crystals/$det.yaml")
    sim = Simulation{T}(LegendData, meta, xtal_meta)

    sim.detector = SolidStateDetector(
        sim.detector,
        contact_id=2,
        contact_potential=meta.characterization.l200_site.recommended_voltage_in_V
    )

    @info "Calculating electric potential..."
    calculate_electric_potential!(
        sim,
        refinement_limits=[0.2, 0.1, 0.05, 0.01],
        depletion_handling=true
    )

    @info "Calculating electric field..."
    calculate_electric_field!(sim, n_points_in_φ=2)

    @info "Calculating weighting potential..."
    calculate_weighting_potential!(
        sim,
        sim.detector.contacts[1].id,
        refinement_limits=[0.2, 0.1, 0.05, 0.01],
        n_points_in_φ=2,
        verbose=false
    )

    function make_axis(T, boundary, gridsize)
        # define interior domain strictly within (0, boundary)
        offset = 2*SSD.ConstructiveSolidGeometry.csg_default_tol(T)
        inner_start = 0 + offset
        inner_stop = boundary - offset

        # compute number of intervals in the interior
        n = round(Int, (inner_stop - inner_start) / gridsize)

        # recompute step to fit the inner domain evenly
        step = (inner_stop - inner_start) / n

        # create interior axis
        axis = range(inner_start, step=step, length=n + 1)

        # prepend and append slightly out-of-bound points
        extended_axis = [0 - offset, axis..., boundary + offset]

        return extended_axis
    end

    gridsize = 0.0005 # in m
    radius = meta.geometry.radius_in_mm / 1000
    height = meta.geometry.height_in_mm / 1000

    x_axis = make_axis(T, radius, gridsize)
    z_axis = make_axis(T, height, gridsize)

    spawn_positions = CartesianPoint{T}[]
    idx_spawn_positions = CartesianIndex[]
    for (i,x) in enumerate(x_axis)
        for (k,z) in enumerate(z_axis)
            push!(spawn_positions, CartesianPoint(T[x,0,z]))
            push!(idx_spawn_positions, CartesianIndex(i,k))
        end
    end
    in_idx = findall(x -> x in sim.detector && !in(x, sim.detector.contacts), spawn_positions)

    # simulate events

    time_step = T(1)u"ns"
    max_nsteps = 10000

    # prepare thread-local storage
    n = length(in_idx)
    wfs_raw_threaded = Vector{Vector{Float64}}(undef, n)
    dt_threaded = Vector{Int}(undef, n)

    @info "Simulating energy depositions on grid r=0:$gridsize:$radius and z=0:$gridsize:$height..."
    @threads for i in 1:n
        p = spawn_positions[in_idx[i]]
        e = SSD.Event([p], [2039u"keV"])
        drift_charges!(e, sim, Δt = time_step, max_nsteps = max_nsteps, verbose = false)

        # store results in preallocated arrays
        lhpath = length(e.drift_paths[1].h_path)
        lepath = length(e.drift_paths[1].e_path)
        dt_threaded[i] = max(lepath, lhpath)
    end

    # assign final results
    dt = dt_threaded

    drift_time = fill(NaN, length(x_axis), length(z_axis))
    for (i, idx) in enumerate(idx_spawn_positions[in_idx])
        drift_time[idx] = dt[i]
    end

        output = (
            r=collect(x_axis) * u"m",
            z=collect(z_axis) * u"m",
            drift_time=transpose(drift_time) * u"ns",
        )
        if id == 1
            mode = "w"
        else
            mode = "r+"
        end

        @info "Saving to disk..."

        lh5open("drift-time-maps.lh5", mode) do f
            f["$(det)A"] = output
        end
end
