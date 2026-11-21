using System;
using System.IO;
using System.Globalization;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Action
{
    // Only enabled on a development worker. Learner, teacher and executed inputs
    // are written separately so that Bellman updates use the executed controls.
    internal sealed class SistersNativeDemonstrator : IDisposable
    {
        private readonly string _folder;
        private readonly SistersTactics? _teacher;
        private readonly Random _random;
        private BinaryWriter? _tape;
        private int _episodes;
        private float _until;
        private bool _teacherActs;
        private readonly double? _teacherRatio;
        private readonly bool _recordTiming;
        private readonly bool _recordPlans;
        private readonly SistersOnlineChoices? _online;
        internal SistersNeuralBlockTrial? NeuralTrial { get; }

        private SistersNativeDemonstrator(string folder, SistersTactics? teacher)
        {
            _folder=folder;
            _teacher=teacher;
            _recordTiming = Environment.GetEnvironmentVariable("HKRL_NATIVE_TIMING") == "1";
            _recordPlans = Environment.GetEnvironmentVariable("HKRL_NATIVE_PLAN_TRACE") == "1";
            var ratio = Environment.GetEnvironmentVariable("HKRL_NATIVE_DEMO_RATIO");
            if (!string.IsNullOrWhiteSpace(ratio))
            {
                if (!double.TryParse(ratio, NumberStyles.Float, CultureInfo.InvariantCulture, out var value) ||
                    double.IsNaN(value) || value < 0 || value > 1) throw new InvalidOperationException("Invalid teacher ratio");
                _teacherRatio = value;
            }
            int.TryParse(Environment.GetEnvironmentVariable("HKRL_NATIVE_DEMO_OFFSET"), out _episodes);
            if (_episodes < 0) throw new InvalidOperationException("Invalid completed demonstration count");
            int seed;
            if (!int.TryParse(Environment.GetEnvironmentVariable("HKRL_NATIVE_DEMO_SEED"), out seed)) seed = 4913;
            _random = new Random(seed);
            if (Environment.GetEnvironmentVariable("HKRL_NATIVE_ONLINE_CHOICES") == "1")
            {
                if (_teacherRatio != 0 || !_recordTiming)
                    throw new InvalidOperationException("Online sampling requires a timing trace and zero teacher control");
                _online = new SistersOnlineChoices(folder, seed);
            }
            var candidate = Environment.GetEnvironmentVariable("HKRL_NATIVE_TRIAL_CANDIDATE");
            if (!string.IsNullOrWhiteSpace(candidate))
            {
                if (teacher != null || _online != null || _recordPlans || !_recordTiming || _teacherRatio != 0)
                    throw new InvalidOperationException("Neural block comparisons require exclusive teacher-free timing recording");
                NeuralTrial = new SistersNeuralBlockTrial(folder, candidate!, seed);
            }
            if (teacher == null && ((_online == null && NeuralTrial == null) || _recordPlans))
                throw new InvalidOperationException("Teacher-free recording requires an online trial without plan labels");
        }

        internal static SistersNativeDemonstrator? FromEnvironment()
        {
            string? folder=Environment.GetEnvironmentVariable("HKRL_NATIVE_DEMO_DIR");
            if (string.IsNullOrWhiteSpace(folder)) return null;
            var teacher=SistersTactics.FromEnvironment();
            if (!Directory.Exists(folder))
                throw new InvalidOperationException("Missing development recording directory");
            return new SistersNativeDemonstrator(folder!,teacher);
        }

        internal void Reset(ulong episode)
        {
            _tape?.Dispose();
            _tape=new BinaryWriter(new FileStream(Path.Combine(_folder,episode+".native"),FileMode.CreateNew,FileAccess.Write,FileShare.Read));
            _tape.Write(_teacher == null ? 0x33474144 : _recordTiming ? 0x32474144 : 0x31474144);
            _tape.Write(episode);
            _until=-1;
            _online?.Reset(episode);
            _episodes++;
        }

        internal bool RecordsTiming => _recordTiming;
        internal bool RecordsRawFeatures => _teacher != null;

        internal PrimitiveInput Label(ObservationSnapshot obs, ulong tick, float[] raw, PrimitiveInput learner, float[]? timing,
            NativeSistersNetwork active)
        {
            float now=obs.Global.TimeInEpisode;
            if (_online != null && timing![SistersTimingState.TrioFlag] > .5f)
                learner = _online.Choose(now, active, learner, timing!);
            var tape=_tape!;
            if (_teacher == null)
            {
                tape.Write(now);
                Write(tape, learner);
                foreach (float x in timing!) tape.Write(x);
                return learner;
            }
            _teacher.Decide(obs,tick);
            var label=_teacher.Input;
            if (now>=_until)
            {
                double teacherProbability=_teacherRatio ?? (_episodes<=4 ? .8 : _episodes<=8 ? .5 : .2);
                _teacherActs=_random.NextDouble()<teacherProbability;
                _until=now+.8f;
            }
            var executed=_teacherActs?label:learner;
            tape.Write(now);
            foreach (float x in raw) tape.Write(x);
            Write(tape,executed);
            Write(tape,label);
            Write(tape,learner);
            tape.Write(_teacherActs?1f:0f);
            if (_recordTiming) foreach (float x in timing!) tape.Write(x);
            return executed;
        }

        private static void Write(BinaryWriter writer, PrimitiveInput a)
        {
            writer.Write((float)a.MovementX);
            writer.Write((float)a.AimY);
            writer.Write((float)a.Buttons);
        }

        internal void Flush() { _tape?.Flush(); _online?.Flush(); NeuralTrial?.Flush(); if (_recordPlans) _teacher!.Flush(); }
        public void Dispose() { _tape?.Dispose(); _online?.Dispose(); NeuralTrial?.Dispose(); _teacher?.Dispose(); }
    }
}
