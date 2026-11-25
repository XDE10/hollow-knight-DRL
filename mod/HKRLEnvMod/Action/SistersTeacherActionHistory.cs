using System;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Action
{
    // Counterfactual choices receive a copy. Only the executed command advances history.
    internal struct SistersTeacherActionHistory
    {
        private const uint JumpTap = 1u << ActionMasker.ButtonJumpTap;
        private const uint JumpHold = 1u << ActionMasker.ButtonJumpHold;
        private const uint Jump = JumpTap | JumpHold;
        private const uint Attack = 1u << ActionMasker.ButtonAttack;
        private const uint Cast = 1u << ActionMasker.ButtonCast;
        private const uint Focus = 1u << ActionMasker.ButtonFocusHold;
        private const uint ArtHold = 1u << ActionMasker.ButtonNailArtHold;
        private const uint ArtRelease = 1u << ActionMasker.ButtonNailArtRelease;
        private const float Inactive = -10f;

        internal float JumpEnd, JumpAt, LastDash, LastAttack, LastCast, HealAt, ArtAt, ArtReleased;
        internal int HealHp, CastAim, LastMove;
        internal bool GroundTakeoff;
        internal uint PreviousButtons;

        internal static SistersTeacherActionHistory Initial => new SistersTeacherActionHistory
        {
            JumpEnd = Inactive, JumpAt = Inactive, LastDash = Inactive, LastAttack = Inactive,
            LastCast = Inactive, HealAt = Inactive, ArtAt = Inactive, ArtReleased = Inactive,
        };

        internal void CommitLearner(in PlayerObservation p, float now, PrimitiveInput executed,
            PrimitiveInput advised, in SistersTeacherActionHistory planned, bool sequence)
        {
            uint buttons = executed.Buttons;
            uint rising = buttons & ~PreviousButtons;
            LastMove = executed.MovementX;

            // The game receives one jump signal for both tap and hold.
            bool jumping = (buttons & Jump) != 0;
            bool missedTakeoff = sequence && GroundTakeoff && p.OnGround &&
                now - JumpAt > .10f && now < JumpEnd;
            if (!p.OnGround || missedTakeoff) GroundTakeoff = false;
            if (!jumping || missedTakeoff)
            {
                if (JumpEnd > now) JumpEnd = now - .06f;
                GroundTakeoff = false;
            }
            else if ((PreviousButtons & Jump) == 0 &&
                (p.OnGround || (p.DoubleJumpAvailable && !p.Dashing)))
            {
                JumpAt = now;
                GroundTakeoff = p.OnGround;
                // Reuse a proposed hold only for the same jump actually requested.
                JumpEnd = (buttons & Jump) == (advised.Buttons & Jump) && planned.JumpAt == now
                    ? planned.JumpEnd
                    : now + ((buttons & JumpHold) != 0 ? .06f : 0f);
            }
            else if ((buttons & JumpHold) != 0 && JumpAt > Inactive && (p.Jumping || p.VelY > 0))
                JumpEnd = Math.Max(JumpEnd, now + .06f);

            // Nail Art and ordinary attack share the physical attack signal.
            if ((rising & Attack) != 0 && (PreviousButtons & ArtHold) == 0 &&
                (buttons & ArtRelease) == 0 && p.CanAttack)
                LastAttack = now;
            if ((rising & Cast) != 0 && p.CanCast && p.Soul >= 33)
            {
                LastCast = now;
                CastAim = executed.AimY;
            }

            if ((buttons & Focus) == 0 || (HealAt >= 0 && (p.Hp != HealHp || now - HealAt > 1.40f)))
                HealAt = Inactive;
            else if ((rising & Focus) != 0 && p.Hp < p.MaxHp && p.Soul >= 33 && p.OnGround && p.CanFocus)
            {
                HealAt = now;
                HealHp = p.Hp;
            }

            if ((buttons & ArtHold) == 0 || (buttons & ArtRelease) != 0)
            {
                if (ArtAt >= 0 && (p.NailChargeTimer >= 1.35f || now - ArtAt >= 1.4f))
                {
                    ArtReleased = now;
                    LastAttack = now;
                }
                ArtAt = Inactive;
            }
            else if ((rising & ArtHold) != 0 && (p.CanNailCharge || p.NailChargeTimer > 0))
                ArtAt = now;
        }
    }
}

