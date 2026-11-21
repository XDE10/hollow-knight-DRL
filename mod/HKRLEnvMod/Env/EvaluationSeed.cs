using System;
using System.Globalization;
using System.IO;

namespace HKRLEnvMod.Env
{
    // Opt-in for a dedicated evaluation process. The first RESET uses baseSeed;
    // each subsequent episode advances it by one before loading the arena.
    internal static class EvaluationSeed
    {
        internal static void Apply(ulong episode)
        {
            string? value = Environment.GetEnvironmentVariable("HKRL_ARENA_BASE_SEED");
            if (string.IsNullOrEmpty(value)) return;
            string? receipt = Environment.GetEnvironmentVariable("HKRL_ARENA_SEED_RECEIPT");
            if (string.IsNullOrEmpty(receipt) ||
                !int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int first))
                throw new InvalidOperationException("Seeded evaluation requires an Int32 seed and receipt path.");
            int seed = checked(first + (int)(episode - 1));
            UnityEngine.Random.InitState(seed);
            File.WriteAllText(receipt, FormattableString.Invariant($"{episode}\n{seed}\n"));
        }
    }
}
