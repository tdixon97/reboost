from collections.abc import Iterable
import reboost.hpge.utils as utils
import reboost.hpge.processors as processors

def build_hit(lh5_in_file: str, lh5_out_file: str, detectors: Iterable[str | int], buffer_len: int = int(5e6))->None:

    for idx,d in enumerate(detectors):
        delete_input = True if (idx==0) else False
        utils.read_write_incremental(lh5_out_file,f"hit/{d}",processors.group_by_event,f"hit/{d}",
                                     lh5_in_file,buffer_len,delete_input=delete_input)
