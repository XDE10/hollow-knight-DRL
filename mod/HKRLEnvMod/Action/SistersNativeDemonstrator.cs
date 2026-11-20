using System;
using System.IO;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Action
{
    // Only enabled on a development worker. Learner, teacher and executed inputs
    // are written separately so that Bellman updates use the executed controls.
    internal sealed class SistersNativeDemonstrator : IDisposable
    {
        private readonly string _folder;
        private readonly SistersTactics _teacher;
        private readonly Random _random;
        private BinaryWriter? _tape;
        private int _episodes;
        private float _until;
        private bool _teacherActs;

        private SistersNativeDemonstrator(string folder, SistersTactics teacher)
        {
            _folder=folder;
            _teacher=teacher;
            int.TryParse(Environment.GetEnvironmentVariable("HKRL_NATIVE_DEMO_OFFSET"), out _episodes);
            if (_episodes < 0) throw new InvalidOperationException("Invalid completed demonstration count");
            int seed;
            if (!int.TryParse(Environment.GetEnvironmentVariable("HKRL_NATIVE_DEMO_SEED"), out seed)) seed = 4913;
            _random = new Random(seed);
        }

        internal static SistersNativeDemonstrator? FromEnvironment()
        {
            string? folder=Environment.GetEnvironmentVariable("HKRL_NATIVE_DEMO_DIR");
            if (string.IsNullOrWhiteSpace(folder)) return null;
            var teacher=SistersTactics.FromEnvironment();
            if (teacher==null || !Directory.Exists(folder))
                throw new InvalidOperationException("Development demonstrations need a teacher and a recording directory");
            return new SistersNativeDemonstrator(folder!,teacher);
        }

        internal void Reset(ulong episode)
        {
            _tape?.Dispose();
            _tape=new BinaryWriter(new FileStream(Path.Combine(_folder,episode+".native"),FileMode.CreateNew,FileAccess.Write,FileShare.Read));
            _tape.Write(0x31474144);
            _tape.Write(episode);
            _until=-1;
            _episodes++;
        }

        internal PrimitiveInput Label(ObservationSnapshot obs, ulong tick, float[] raw, PrimitiveInput learner)
        {
            float now=obs.Global.TimeInEpisode;
            _teacher.Decide(obs,tick);
            var label=_teacher.Input;
            if (now>=_until)
            {
                double teacherProbability=_episodes<=4 ? .8 : _episodes<=8 ? .5 : .2;
                _teacherActs=_random.NextDouble()<teacherProbability;
                _until=now+.8f;
            }
            var executed=_teacherActs?label:learner;
            var tape=_tape!;
            tape.Write(now);
            foreach (float x in raw) tape.Write(x);
            Write(tape,executed);
            Write(tape,label);
            Write(tape,learner);
            tape.Write(_teacherActs?1f:0f);
            return executed;
        }

        private static void Write(BinaryWriter writer, PrimitiveInput a)
        {
            writer.Write((float)a.MovementX);
            writer.Write((float)a.AimY);
            writer.Write((float)a.Buttons);
        }

        internal void Flush() { _tape?.Flush(); }
        public void Dispose() { _tape?.Dispose(); _teacher.Dispose(); }
    }
}
