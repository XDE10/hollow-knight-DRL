using System;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Rewards
{
    internal static class SistersNailCurriculum
    {
        private static int _multiplier = 1;

        internal static void Install(int port, int saveSlot)
        {
            string? configured = Environment.GetEnvironmentVariable("HKRL_SISTERS_NAIL_MULTIPLIER");
            if (string.IsNullOrEmpty(configured) || configured == "1")
                return;

            if (port == 5555 || saveSlot == 3
                || !int.TryParse(configured, out int multiplier)
                || multiplier < 2 || multiplier > 8)
                throw new InvalidOperationException("Sisters nail curriculum requires a training copy and a multiplier in [2, 8].");

            _multiplier = multiplier;
            On.HealthManager.Hit += OnHit;
            Debug.Logger.Info($"Sisters training nail multiplier: {_multiplier}.");
        }

        internal static void Uninstall()
        {
            if (_multiplier != 1)
                On.HealthManager.Hit -= OnHit;
            _multiplier = 1;
        }

        private static void OnHit(On.HealthManager.orig_Hit orig, HealthManager boss, HitInstance hit)
        {
            try
            {
                if (hit.AttackType == global::AttackTypes.Nail && hit.DamageDealt > 0
                    && UnityEngine.SceneManagement.SceneManager.GetActiveScene().name == SistersContact.Scene)
                {
                    foreach (HealthManager active in BossLocator.FindActiveBosses())
                    {
                        if (active != boss)
                            continue;
                        hit.DamageDealt = checked(hit.DamageDealt * _multiplier);
                        break;
                    }
                }
            }
            catch (Exception exception)
            {
                Debug.Logger.Error("Failed to apply Sisters nail curriculum", exception);
            }

            orig(boss, hit);
        }
    }
}
