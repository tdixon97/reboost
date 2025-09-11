from __future__ import annotations


def print_random_crash_msg(rng):
    msgs = [
        "Segmentation fault (core dumped)",
        "zsh: segmentation fault ./orca.out",
        "Segmentation fault: 11",
        "Bus error (core dumped)",
        "Bus error: 10",
        "*** stack smashing detected ***: terminated",
        "free(): double free detected in tcache 2",
        "free(): invalid pointer",
        "malloc(): corrupted top size",
        "malloc(): memory corruption",
        "malloc(): unaligned tcache chunk detected",
        "Killed",
        "Out of memory: Killed process 4321 (orca.out)",
        "Illegal instruction (core dumped)",
        "Illegal instruction: 4",
        "general protection fault: 0000 [#1] SMP",
        "fish: Job 1, './orca.out' terminated by signal SIGSEGV (Address boundary error)",
        "==24567==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000010",
        "Abort trap: 6",
        "Trace/BPT trap: 5",
    ]

    print(msgs[rng.choice(len(msgs))])  # noqa: T201
