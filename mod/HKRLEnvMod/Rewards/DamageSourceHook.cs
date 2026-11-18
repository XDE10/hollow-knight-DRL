using System;
using HKRLEnvMod.Observation;
using UnityEngine;

namespace HKRLEnvMod.Rewards
{
    internal static class DamageSourceHook
    {
        private static EntityRegistry? _registry;
        private static int _nestedLoss;
        private static readonly bool TraceSources = Environment.GetEnvironmentVariable("HKRL_TRACE_CONTACTS") == "1";
        private static readonly System.Collections.Generic.HashSet<string> Traced = new();

        internal static void Install(EntityRegistry registry)
        {
            Uninstall();
            _registry = registry;
            On.HeroController.TakeDamage += OnContact;
        }

        internal static void Uninstall()
        {
            On.HeroController.TakeDamage -= OnContact;
            _registry = null;
        }

        private static void OnContact(On.HeroController.orig_TakeDamage orig, HeroController hero,
            GameObject hit, GlobalEnums.CollisionSide side, int amount, int hazardType)
        {
            var data = PlayerData.instance;
            if (data == null)
            {
                orig(hero, hit, side, amount, hazardType);
                return;
            }
            int before = data.health + data.healthBlue;
            int parentLoss = _nestedLoss;
            _nestedLoss = 0;
            DamageHooks.ContactDepth++;
            try
            {
                orig(hero, hit, side, amount, hazardType);
            }
            finally
            {
                DamageHooks.ContactDepth--;
                int lost = Math.Max(0, before - data.health - data.healthBlue);
                int ownLoss = lost - _nestedLoss;
                _nestedLoss = DamageHooks.ContactDepth == 0 ? 0 : parentLoss + lost;
                if (ownLoss > 0)
                {
                    try
                    {
                        var source = SistersContact.Resolve(hit, out var owner);
                        if (TraceSources)
                        {
                            string path = "";
                            for (Transform? node = hit == null ? null : hit.transform; node != null; node = node.parent)
                                path = "/" + node.gameObject.name + path;
                            string description = $"Sisters contact {source}: scene={hit?.scene.name}, path={path}, hazard={hazardType}";
                            if (Traced.Add(description)) global::HKRLEnvMod.Debug.Logger.Info(description);
                        }
                        int id = owner != null && source != HKRL.DamageSource.Unknown
                            ? _registry?.GetStableId(owner.GetInstanceID()) ?? 0 : 0;
                        DamageHooks.RecordDamageTaken(id, ownLoss, hazardType, (int)source);
                    }
                    catch (Exception error)
                    {
                        DamageHooks.RecordDamageTaken(0, ownLoss, hazardType);
                        global::HKRLEnvMod.Debug.Logger.Error("Failed to resolve damage source", error);
                    }
                }
            }
        }
    }
}
