using System;
using System.IO;
using System.Reflection;
using HKRLEnvMod.Action;
using HKRLEnvMod.Observation;

internal static class SistersTeacherHistorySmoke
{
    private static uint Bit(int button) => 1u << button;

    private static void Check(bool pass, string message)
    {
        if (!pass) throw new InvalidOperationException(message);
    }

    private static SistersTeacherActionHistory History(SistersTactics teacher) =>
        (SistersTeacherActionHistory)typeof(SistersTactics)
            .GetField("_history", BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(teacher)!;

    private static bool Same(PrimitiveInput a, PrimitiveInput b) =>
        a.Buttons == b.Buttons && a.MovementX == b.MovementX && a.AimY == b.AimY;

    internal static void Run()
    {
        uint jump = Bit(ActionMasker.ButtonJumpTap) | Bit(ActionMasker.ButtonJumpHold);
        uint attack = Bit(ActionMasker.ButtonAttack), cast = Bit(ActionMasker.ButtonCast);
        uint focus = Bit(ActionMasker.ButtonFocusHold);
        uint art = Bit(ActionMasker.ButtonNailArtHold), release = Bit(ActionMasker.ButtonNailArtRelease);
        var p = new PlayerObservation
        {
            PosX = 30, PosY = 7.41f, Hp = 7, MaxHp = 9, Soul = 66, Facing = -1,
            OnGround = true, CanAttack = true, CanFocus = true, CanCast = true, CanNailCharge = true,
        };
        var actual = SistersTeacherActionHistory.Initial;
        var planned = actual;
        planned.JumpAt = planned.LastAttack = planned.LastCast = 1;
        planned.JumpEnd = 1.3f;
        planned.LastMove = 1;
        planned.CastAim = -1;
        var proposal = new PrimitiveInput(1, -1, jump | attack | cast);
        actual.CommitLearner(p, 1, PrimitiveInput.Noop, proposal, planned, true);
        Check(actual.JumpAt == -10 && actual.LastAttack == -10 && actual.LastCast == -10,
            "Noop must not acquire the proposed jump, cast or attack timers");
        var foreign = new PrimitiveInput(-1, 1, cast | jump);
        actual.CommitLearner(p, 1, foreign, proposal, planned, true);
        actual.PreviousButtons = foreign.Buttons;
        Check(actual.JumpAt == 1 && actual.LastCast == 1 && actual.LastAttack == -10 &&
            actual.CastAim == 1 && actual.LastMove == -1, "student aim and movement must drive history");
        actual.CommitLearner(p with { OnGround = false }, 1.04f, PrimitiveInput.Noop, proposal, planned, true);
        Check(!actual.GroundTakeoff && actual.JumpEnd < 1.04f, "jump release ends a proposed hold");

        actual = SistersTeacherActionHistory.Initial;
        var charge = new PrimitiveInput(0, 0, art);
        actual.CommitLearner(p, 3, charge, PrimitiveInput.Noop, planned, true);
        actual.PreviousButtons = art;
        actual.CommitLearner(p with { NailChargeTimer = 1.45f }, 4.45f,
            new PrimitiveInput(0, 0, release), PrimitiveInput.Noop, planned, true);
        Check(actual.ArtAt == -10 && actual.ArtReleased == 4.45f && actual.LastAttack == 4.45f,
            "actual charged release is recorded");

        string folder = Path.Combine(Path.GetTempPath(), "hkrl-teacher-" + Guid.NewGuid());
        Directory.CreateDirectory(folder);
        string config = Path.Combine(folder, "teacher.txt");
        File.WriteAllText(config, "mode=1\nheal=1\ncast=0\ndash=0\nsequence=1");
        string? previous = Environment.GetEnvironmentVariable("HKRL_TACTICS_POLICY");
        Environment.SetEnvironmentVariable("HKRL_TACTICS_POLICY", config);
        try
        {
            using var direct = SistersTactics.FromEnvironment()!;
            using var executing = SistersTactics.FromEnvironment()!;
            using var matching = SistersTactics.FromEnvironment()!;
            using var differing = SistersTactics.FromEnvironment()!;
            var boss = new EntityObservation(1, HKRL.EntityType.Boss, HKRL.Team.Enemy,
                1928179328, 32, 7.41f, 0, 0, 32, 7.41f, 2, 2, 500, 750);
            var frame = new ObservationSnapshot(new GlobalObservation(1, 1), p, new[] { boss });
            for (ulong tick = 1; tick <= 4; tick++)
            {
                frame = frame with { Global = new GlobalObservation(1, 1 + (tick - 1) * .04f) };
                direct.Decide(frame, tick);
                var teacherKey = executing.Demonstrate(frame, tick, PrimitiveInput.Noop, true);
                Check(Same(direct.Input, teacherKey) && (tick >= 3 || (teacherKey.Buttons & attack) != 0),
                    "executing teacher retains the direct attack path");
                var label = matching.Demonstrate(frame, tick, teacherKey, false);
                Check(Same(label, teacherKey), "executed matching keys agree on the next teacher decision");
                if (tick <= 2)
                {
                    differing.Demonstrate(frame, tick, new PrimitiveInput(-1, 0, 0), false);
                    Check(History(differing).LastAttack == -10 && History(differing).LastMove == -1,
                        "discarded counterfactual cannot extend attack latch");
                }
            }
            var attackUnavailable = frame with
            {
                Player = p with { CanAttack = false }, Global = new GlobalObservation(1, 1.06f),
            };
            using var missingAttack = SistersTactics.FromEnvironment()!;
            missingAttack.Demonstrate(frame with { Global = new GlobalObservation(1, 1) }, 1,
                PrimitiveInput.Noop, false);
            Check((missingAttack.Demonstrate(attackUnavailable, 2, PrimitiveInput.Noop, false).Buttons & attack) == 0,
                "a discarded attack cannot create a second-frame pulse");

            var rest = new ObservationSnapshot(new GlobalObservation(2, 2), p, Array.Empty<EntityObservation>());
            Check((differing.Demonstrate(rest, 1, PrimitiveInput.Noop, false).Buttons & focus) != 0 &&
                History(differing).HealAt == -10, "teacher Focus is advisory until executed");
            var holdingFocus = new PrimitiveInput(0, 0, focus);
            differing.Demonstrate(rest with { Global = new GlobalObservation(2, 2.04f) }, 2, holdingFocus, false);
            differing.Demonstrate(rest with
            {
                Global = new GlobalObservation(2, 2.08f), Player = p with { CanFocus = false },
            }, 3, holdingFocus, false);
            Check(History(differing).HealAt == 2.04f, "Focus holds keep their original edge");
            differing.Demonstrate(rest with { Global = new GlobalObservation(2, 3.50f) }, 4, holdingFocus, false);
            Check(History(differing).HealAt == -10, "Focus expires after its hold limit");
            differing.Demonstrate(rest with { Global = new GlobalObservation(2, 3.54f) }, 5, holdingFocus, false);
            Check(History(differing).HealAt == -10, "held Focus cannot be mistaken for a fresh press");
            differing.Demonstrate(rest with { Global = new GlobalObservation(2, 3.58f) }, 6, PrimitiveInput.Noop, false);
            differing.Demonstrate(rest with { Global = new GlobalObservation(2, 3.62f) }, 7, holdingFocus, false);
            Check(History(differing).HealAt == 3.62f, "fresh Focus press has its own timer");
            differing.Demonstrate(rest with { Global = new GlobalObservation(2, 3.66f) }, 8, PrimitiveInput.Noop, false);
            Check(History(differing).HealAt == -10, "released Focus cannot persist");
        }
        finally
        {
            Environment.SetEnvironmentVariable("HKRL_TACTICS_POLICY", previous);
            Directory.Delete(folder, true);
        }
    }
}

namespace HKRLEnvMod.Observation
{
    internal readonly record struct GlobalObservation(ulong EpisodeId, float TimeInEpisode);
    internal readonly record struct ObservationSnapshot(GlobalObservation Global,
        PlayerObservation Player, EntityObservation[] Entities);
    internal readonly record struct EntityObservation(int EntityId, HKRL.EntityType EntityType, HKRL.Team Team,
        int FsmStateHash, float PosX, float PosY, float VelX, float VelY,
        float HurtboxCenterX, float HurtboxCenterY, float HurtboxSizeX, float HurtboxSizeY, int Hp, int MaxHp);
    internal readonly record struct PlayerObservation
    {
        public float PosX { get; init; }
        public float PosY { get; init; }
        public float VelX { get; init; }
        public float VelY { get; init; }
        public float NailChargeTimer { get; init; }
        public int Hp { get; init; }
        public int MaxHp { get; init; }
        public int Soul { get; init; }
        public int Facing { get; init; }
        public bool OnGround { get; init; }
        public bool Jumping { get; init; }
        public bool Dashing { get; init; }
        public bool ShadowDashing { get; init; }
        public bool DoubleJumpAvailable { get; init; }
        public bool Invulnerable { get; init; }
        public bool CanAttack { get; init; }
        public bool CanCast { get; init; }
        public bool CanDash { get; init; }
        public bool CanFocus { get; init; }
        public bool CanNailCharge { get; init; }
    }
}

namespace HKRL
{
    internal enum EntityType { Boss, Projectile }
    internal enum Team { Enemy }
}

namespace HKRLEnvMod.Action
{
    internal interface ISistersLocalActor : IDisposable { }
    // All test scenarios disable the optional heads.
    internal sealed class SistersPolicyHead : IDisposable
    {
        public void Reset(string path, ulong id, float policyWeight, float worldWeight) { }
        public void Observe(ObservationSnapshot obs) { }
        public void BeginChoice() { }
        public float Preference(int move, bool jump, bool dash) => 0;
        public void Compare(float cost, int move, bool jump, bool dash) { }
        public void FinalizeChoice(int move, bool jump, bool dash) { }
        public void Record(float time, PrimitiveInput input) { }
        public void Flush() { }
        public void Dispose() { }
    }
}

namespace UnityEngine
{
    internal static class QualitySettings { internal static int vSyncCount; }
    internal static class Application { internal static int targetFrameRate; }
}

