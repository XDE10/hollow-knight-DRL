using System;
using UnityEngine;

namespace HKRLEnvMod.Observation
{
    internal static class SistersContact
    {
        internal const string Scene = "GG_Mantis_Lords_V";

        // Live Godhome discs are spawned in Unity's persistent object pool.
        internal static bool InSistersPool(GameObject obj)
        {
            return obj != null && obj.scene.name == "DontDestroyOnLoad"
                && UnityEngine.SceneManagement.SceneManager.GetActiveScene().name == Scene;
        }

        // Resolve a collider to one logical actor. All instances of a prefab keep
        // separate Unity IDs. A launched blade can still be parented to its owner.
        internal static HKRL.DamageSource Resolve(GameObject hit, out GameObject owner)
        {
            owner = hit;
            bool pooled = InSistersPool(hit);
            if (hit == null || (hit.scene.name != Scene && !pooled))
                return HKRL.DamageSource.Unknown;

            for (Transform node = hit.transform; node != null; node = node.parent)
            {
                var obj = node.gameObject;
                var name = obj.name;
                if (pooled)
                {
                    if (!name.StartsWith("Shot Mantis Lord", StringComparison.OrdinalIgnoreCase)) continue;
                    owner = obj;
                    return HKRL.DamageSource.EnemyProjectile;
                }
                if (name.StartsWith("Cave Spikes Invis", StringComparison.Ordinal))
                {
                    owner = obj;
                    return HKRL.DamageSource.Spikes;
                }
                if (name.StartsWith("Shot Mantis Lord", StringComparison.OrdinalIgnoreCase)
                    || name.IndexOf("projectile", StringComparison.OrdinalIgnoreCase) >= 0
                    || name.IndexOf("boomerang", StringComparison.OrdinalIgnoreCase) >= 0)
                {
                    owner = obj;
                    return HKRL.DamageSource.EnemyProjectile;
                }
                if (obj.GetComponent<global::HealthManager>() != null
                    && name.StartsWith("Mantis Lord", StringComparison.OrdinalIgnoreCase))
                {
                    owner = obj;
                    return HKRL.DamageSource.BossBody;
                }
            }
            return HKRL.DamageSource.Unknown;
        }
    }
}
