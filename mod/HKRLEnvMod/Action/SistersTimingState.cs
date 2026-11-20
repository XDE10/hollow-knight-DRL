using System;
using System.Collections.Generic;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Action
{
    // Causal measurements shared by the development recorder and frozen actor.
    // No tactical controller, upcoming input labels or future game events are read.
    internal sealed class SistersTimingState
    {
        internal const int Hero = 24, Boss = 36, Disc = 14, Width = 192;
        internal const int TrioFlag = 23;
        private static readonly int[] States = {
            289970528, 291868896, -1523407744, -1376807296, -737267072,
            -1374908928, 1928179328, 907425216, -1403342080, 739257600,
            -1368652544, 1871902336, 735357952, 157285984, 7946978, 1756965248
        };
        private struct Track
        {
            internal int State;
            internal float Since, EntryDx, EntryVx, Last, Vx, Ax;
        }
        private readonly Dictionary<int, Track> _tracks = new();
        private readonly float[] _bossDistance = new float[3], _discDistance = new float[2];
        private readonly float[] _pressed = new float[9], _released = new float[9];
        private float _hurtAt = -100, _moveAt = -100, _aimAt = -100;
        private int _hp;
        private bool _trio;
        private PrimitiveInput _prior;
        internal readonly float[] Values = new float[Width];

        internal SistersTimingState()
        {
            for (int i = 0; i < 9; i++) { _pressed[i] = -100; _released[i] = -100; }
        }
        private static float Limit(float x) => Math.Max(-2f, Math.Min(2f, x));
        private static float Age(float now, float then) => Math.Max(0f, Math.Min(2f, now - then));

        private static int Insert(float[] values, float[] distances, int start, int width, float distance)
        {
            int slot = 0;
            while (slot < distances.Length && distances[slot] <= distance) slot++;
            if (slot == distances.Length) return -1;
            for (int j = distances.Length - 1; j > slot; j--)
            {
                distances[j] = distances[j - 1];
                Array.Copy(values, start + (j - 1) * width, values, start + j * width, width);
            }
            distances[slot] = distance;
            int index = start + slot * width;
            Array.Clear(values, index, width);
            return index;
        }

        internal void Observe(ObservationSnapshot obs)
        {
            Array.Clear(Values, 0, Width);
            for (int j = 0; j < 3; j++) _bossDistance[j] = float.MaxValue;
            for (int j = 0; j < 2; j++) _discDistance[j] = float.MaxValue;
            var p = obs.Player;
            float now = obs.Global.TimeInEpisode;
            if (_hp > p.Hp) _hurtAt = now;
            _hp = p.Hp;
            var v = Values;
            v[0] = Limit((p.PosX - 30) / 10); v[1] = Limit((p.PosY - 7.41f) / 8);
            v[2] = Limit(p.VelX / 15); v[3] = Limit(p.VelY / 20);
            v[4] = p.Hp / (float)Math.Max(1, p.MaxHp); v[5] = p.Soul / 99f;
            v[6] = -p.Facing; v[7] = p.OnGround ? 1 : 0;
            v[8] = p.DoubleJumpAvailable ? 1 : 0; v[9] = p.Invulnerable ? 1 : 0;
            v[10] = p.CanAttack ? 1 : 0; v[11] = p.CanCast ? 1 : 0;
            v[12] = p.CanFocus ? 1 : 0; v[13] = p.CanDash ? 1 : 0;
            v[14] = p.Dashing ? 1 : 0; v[15] = p.ShadowDashing ? 1 : 0;
            v[16] = Limit(p.AttackLockTimer); v[17] = Limit(p.CastLockTimer);
            v[18] = Limit(p.NailChargeTimer); v[19] = Limit(p.DashCooldown);
            v[20] = Limit(p.InvulnTimer); v[21] = Age(now, _hurtAt);
            v[22] = Math.Min(1, now / 6);
            for (int i = 0; i < obs.Entities.Count; i++)
            {
                if (i >= obs.EntityMask.Count || !obs.EntityMask[i]) continue;
                var e = obs.Entities[i];
                bool boss = e.EntityType == HKRL.EntityType.Boss;
                if (!boss && !(e.EntityType == HKRL.EntityType.Projectile && e.Team == HKRL.Team.Enemy)) continue;
                if (boss && e.MaxHp == 750) _trio = true;
                bool box = e.HurtboxSizeX > 0 && e.HurtboxSizeY > 0;
                float x = box ? e.HurtboxCenterX : e.PosX, y = box ? e.HurtboxCenterY : e.PosY;
                float dx = x - p.PosX, dy = y - p.PosY;
                int state = unchecked((int)(float)e.FsmStateHash);
                bool known = _tracks.TryGetValue(e.EntityId, out var t);
                if (!known || t.State != state || now - t.Last > .5f)
                {
                    t.State = state; t.Since = now; t.EntryDx = p.PosX - e.PosX; t.EntryVx = p.VelX;
                }
                float dt = now - t.Last;
                if (known && dt > .005f && dt < .4f) t.Ax = Limit((e.VelX - t.Vx) / dt / 40f);
                else t.Ax = 0;
                t.Vx = e.VelX; t.Last = now;
                _tracks[e.EntityId] = t;

                float distance = Math.Abs(dx) + .65f * Math.Abs(dy);
                int k = boss ? Insert(v, _bossDistance, Hero, Boss, distance)
                             : Insert(v, _discDistance, Hero + 3 * Boss, Disc, distance);
                if (k < 0) continue;
                v[k] = 1; v[k+1] = Limit(dx / 10); v[k+2] = Limit(dy / 8);
                v[k+3] = Limit(e.VelX / (boss ? 55 : 34)); v[k+4] = Limit(e.VelY / (boss ? 65 : 5));
                v[k+5] = Limit((e.PosX - 30) / 10); v[k+6] = Limit((e.PosY - 7.41f) / 9);
                v[k+7] = Limit(e.HurtboxSizeX / 5); v[k+8] = Limit(e.HurtboxSizeY / 5);
                v[k+9] = box ? 1 : 0; v[k+10] = Age(now, t.Since);
                if (boss)
                {
                    v[k+11] = Limit(t.EntryDx / 9); v[k+12] = Limit(t.EntryVx / 15);
                    v[k+13] = e.Hp / (float)Math.Max(1, e.MaxHp);
                    v[k+14] = e.MaxHp == 750 ? 1 : 0;
                    v[k+15] = Limit((e.PosX - p.PosX) / 8); v[k+16] = Limit((e.PosY - p.PosY) / 8);
                    v[k+17] = e.HitboxActive ? 1 : 0;
                    v[k+18] = Limit((e.VelX - p.VelX) / 30);
                    v[k+19] = Limit((e.VelY - p.VelY) / 30);
                    int phase = Array.IndexOf(States, state);
                    if (phase >= 0) v[k+20+phase] = 1;
                }
                else
                {
                    v[k+11] = t.Ax; v[k+12] = Limit((e.VelX - p.VelX) / 34);
                    v[k+13] = Limit((e.VelY - p.VelY) / 20);
                }
            }
            v[TrioFlag] = _trio ? 1 : 0;
            int c = Hero + 3 * Boss + 2 * Disc;
            v[c] = _prior.MovementX; v[c+1] = _prior.AimY;
            v[c+2] = Age(now, _moveAt); v[c+3] = Age(now, _aimAt);
            for (int j = 0; j < 9; j++)
            {
                v[c+4+j] = (_prior.Buttons & (1u << j)) != 0 ? 1 : 0;
                v[c+13+j] = Age(now, _pressed[j]); v[c+22+j] = Age(now, _released[j]);
            }
            v[c+31] = p.CanNailCharge ? 1 : 0;
        }

        internal void Commit(PrimitiveInput input, float time)
        {
            if (input.MovementX != _prior.MovementX) _moveAt = time;
            if (input.AimY != _prior.AimY) _aimAt = time;
            for (int j = 0; j < 9; j++)
            {
                uint mask = 1u << j;
                if (((input.Buttons ^ _prior.Buttons) & mask) == 0) continue;
                if ((input.Buttons & mask) == 0) _released[j] = time;
                else _pressed[j] = time;
            }
            _prior = input;
        }
    }
}
