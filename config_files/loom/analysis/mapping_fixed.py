EXPERIMENT_MANIFEST = [
    # ==========================================
    # 1. BASELINES
    # ==========================================
    {
        "name":   "baseline_buckets_mvd",
        "run_id": "dx7k84zu",
        "folder": "2026-01-13__16-31-27_68708f8c380c52c4",
    },
    {
        "name":   "baseline_isoflop_to_L1024G512memory_and_3loops",
        "run_id": "acw5j6ox",
        "folder": "2026-01-26__14-49-40_02da07efbc8445e6",
    },
    {
        "name":   "baseline_isoparam_to_L1024G512memory",
        "run_id": "zxwh48wu",
        "folder": "2026-01-26__11-06-54_e3e769aaa0033a7e",
    },

    # ==========================================
    # 2. LOOP 1 VARIANTS (Memory Size Scaling)
    # ==========================================
    {
        "name":   "loop1_L512G1024",
        "run_id": "dy705ftt",
        "folder": "2026-01-26__13-52-26_a8c42b25b4d84f03",
    },
    {
        "name":   "loop1_L512G4096",
        "run_id": "1ket85g0",
        "folder": "2026-01-26__11-22-50_82a1729fd64c49f8",
    },
    {
        "name":   "loop1_L1024G512",
        "run_id": "2h4fgihj",
        "folder": "2026-01-26__11-16-53_96a467fd7a75f1db",
    },
    {
        "name":   "loop1_L4096G512",
        "run_id": "jhtbyhil",
        "folder": "2026-01-26__13-49-58_9ffb0931b75711c2",
    },

    # ==========================================
    # 3. LOOP 3 VARIANTS (Core & MVD)
    # ==========================================
    {
        "name":   "loop3_buckets_mvd",
        "run_id": "6kibshcj",
        "folder": "2026-01-15__09-28-10_6293597212cdaed8",
    },
    {
        "name":   "loop3_iso_mvd",
        "run_id": "90ndflhs",
        "folder": "2026-01-13__16-17-05_cfb2b82b2d1f3129",
    },
    {
        "name":   "loop3_L1024G512_cyclical",
        "run_id": "b7l63bce",
        "folder": "2026-01-27__14-04-59_c8fdbd942cf02ca0",
    },
    {
        "name":   "loop3_L1024G512_ponder001",
        "run_id": "7mzvjajg",
        "folder": "2026-01-27__15-24-04_043e421a1b54accb",
    },
    {
        "name":   "loop3_L1024G512_ponder-001",
        "run_id": "a8wysxso",
        "folder": "2026-01-27__15-23-57_938b4efba13f5ed5",
    },

    # ==========================================
    # 4. LOOP 3 VARIANTS (Individual Memory / Init)
    # ==========================================
    {
        "name":   "loop3_L1024G512_individualMemory_frozenmem",
        "run_id": "t5qfh6mu",
        "folder": "2026-01-25__01-09-00_6756580fe4bb252b",
    },
    {
        "name":   "loop3_L1024G512_individualMemory_init0",
        "run_id": "fu2lz8ci",
        "folder": "2026-01-23__22-42-46_6756580fe4bb252b",
    },
    {
        "name":   "loop3_L1024G512_individualMemory_init3",
        "run_id": "89w112fv",
        "folder": "2026-01-23__22-45-27_6756580fe4bb252b",
    },
    {
        "name":   "loop3_L1024G512_individualMemory_init-3",
        "run_id": "zhvqr1sl",
        "folder": "2026-01-23__19-52-07_6756580fe4bb252b",
    },

    # ==========================================
    # 5. HIGH LOOPS (5, 7, 9)
    # ==========================================
    {
        "name":   "loop5_L1024G512",
        "run_id": "vvkqwmg6",
        "folder": "2026-01-27__21-51-37_ad8de814eeeade84",
    },
    {
        "name":   "loop5_buckets_mvd",
        "run_id": "8l32ixdg",
        "folder": "2026-01-13__16-15-53_ac0fe16b31e73093",
    },
    {
        "name":   "loop5_iso_mvd",
        "run_id": "zrgz6gwd",
        "folder": "2026-01-13__16-19-19_4fde1337716312ed",
    },
    {
        "name":   "loop7_buckets_mvd",
        "run_id": "6533grrb",
        "folder": "2026-01-14__08-46-21_d2cfb3a1e9e8f3b1",
    },
    {
        "name":   "loop9_buckets",
        "run_id": "lp4f3bc5",
        "folder": "2026-01-27__18-44-29_54e21865ce16b752",
    },
]