using System;
using System.Globalization;
using System.IO;
using System.Security.Cryptography;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Action
{
    internal interface ISistersLocalActor : IDisposable
    {
        PrimitiveInput Input { get; }
        bool NeedsObservation(ulong tick, ulong episode, ulong commits);
        void Decide(ObservationSnapshot obs, ulong tick);
        void Record(float time);
        void Flush();
    }

    // Direct learned primitive control. The heartbeat only bounds how long a disconnected
    // actor may run. The tactical controller is mutually exclusive with this actor.
    internal sealed class SistersNativeActor : ISistersLocalActor
    {
        private readonly string _path;
        private readonly string _ledger;
        private readonly SistersNativeDemonstrator? _demonstrator;
        private readonly float[] _raw = new float[56];
        private readonly float[] _boss = new float[3];
        private readonly float[] _disc = new float[2];
        private NativeSistersNetwork? _net;
        private ulong _episode, _tick, _commits;
        private int _count, _attack, _cast, _dash, _art;
        private string _digest = "";
        public PrimitiveInput Input { get; private set; }

        private SistersNativeActor(string path, string ledger, SistersNativeDemonstrator? demonstrator)
        {
            _path = path;
            _ledger = ledger;
            _demonstrator = demonstrator;
        }

        public static ISistersLocalActor? FromEnvironment()
        {
            var policy = Environment.GetEnvironmentVariable("HKRL_NATIVE_ACTOR");
            if (string.IsNullOrWhiteSpace(policy)) return SistersTactics.FromEnvironment();
            var demonstrator = SistersNativeDemonstrator.FromEnvironment();
            if (demonstrator == null && !string.IsNullOrWhiteSpace(Environment.GetEnvironmentVariable("HKRL_TACTICS_POLICY")))
                throw new InvalidOperationException("Select only one local Sisters actor");
            var ledger = Environment.GetEnvironmentVariable("HKRL_NATIVE_ACTOR_LEDGER");
            if (string.IsNullOrWhiteSpace(ledger)) throw new InvalidOperationException("Missing native actor ledger path");
            if (Environment.GetEnvironmentVariable("HKRL_COMPACT_RENDER") == "1")
            {
                UnityEngine.QualitySettings.vSyncCount = 0;
                UnityEngine.Application.targetFrameRate = 120;
            }
            return new SistersNativeActor(policy!, ledger!, demonstrator);
        }

        public bool NeedsObservation(ulong tick, ulong episode, ulong commits)
        {
            if (_net != null && episode == _episode && (tick-_tick < 2 || commits == _commits)) return false;
            _commits = commits;
            return true;
        }

        public void Decide(ObservationSnapshot obs, ulong tick)
        {
            if (_net == null || _episode != obs.Global.EpisodeId)
            {
                var bytes = File.ReadAllBytes(_path);
                using (var sha = SHA256.Create())
                    _digest = BitConverter.ToString(sha.ComputeHash(bytes)).Replace("-", "").ToLowerInvariant();
                using (var reader = new BinaryReader(new MemoryStream(bytes)))
                    _net = new NativeSistersNetwork(reader);
                _episode = obs.Global.EpisodeId;
                _count = _attack = _cast = _dash = _art = 0;
                _demonstrator?.Reset(_episode);
            }
            _tick = tick;
            SistersPolicyHead.WriteFeatures(obs, _raw, _boss, _disc);
            Input = _net.Choose(_raw, obs.Global.TimeInEpisode);
            if (_demonstrator != null) Input = _demonstrator.Label(obs, tick, _raw, Input);
            _net.Commit(Input, obs.Global.TimeInEpisode);
            _count++;
            if ((Input.Buttons & 8) != 0) _attack++;
            if ((Input.Buttons & 16) != 0) _cast++;
            if ((Input.Buttons & 4) != 0) _dash++;
            if ((Input.Buttons & 384) != 0) _art++;
        }

        public void Record(float time) { }

        public void Flush()
        {
            _demonstrator?.Flush();
            if (_count == 0) return;
            File.WriteAllText(_ledger, string.Join("\n", _episode.ToString(CultureInfo.InvariantCulture),
                _digest, _count, _attack, _cast, _dash, _art));
        }

        public void Dispose() { _demonstrator?.Dispose(); }
    }

    internal sealed class NativeSistersNetwork
    {
        private const int Physical = 56, Width = 180, Hidden1 = 160, Hidden2 = 128;
        private static readonly uint[] Pulses = { 1, 4, 8, 16, 32, 256 };
        private static readonly uint[] Holds = { 2, 128, 32 };
        private readonly PrimitiveInput[] _actions;
        private readonly float[][] _w = new float[6][];
        private readonly float[] _history = new float[Physical * 4];
        private readonly float[] _x = new float[Width], _h = new float[Hidden1], _z = new float[Hidden2];
        private readonly float[] _scores, _pulseAt = new float[6], _holdAt = new float[3];
        private uint _previous;
        private int _index;

        internal NativeSistersNetwork(BinaryReader reader)
        {
            if (reader.ReadInt32() != 0x314C5149 || reader.ReadInt32() != Width ||
                reader.ReadInt32() != Hidden1 || reader.ReadInt32() != Hidden2)
                throw new InvalidDataException("Invalid native actor architecture");
            int n = reader.ReadInt32();
            if (n < 1 || n > 4096) throw new InvalidDataException("Invalid action count");
            _scores = new float[n];
            _actions = new PrimitiveInput[n];
            for (int i=0; i<n; i++)
            {
                int move=reader.ReadInt32(), aim=reader.ReadInt32(), buttons=reader.ReadInt32();
                if (Math.Abs(move)>1 || Math.Abs(aim)>1 || buttons<0 || buttons>511)
                    throw new InvalidDataException("Invalid primitive in native actor catalogue");
                _actions[i]=new PrimitiveInput(move,aim,(uint)buttons);
            }
            int[] sizes = {Width*Hidden1, Hidden1, Hidden1*Hidden2, Hidden2, Hidden2*n, n};
            for (int layer=0; layer<sizes.Length; layer++)
            {
                var weights = new float[sizes[layer]];
                for (int i=0; i<weights.Length; i++)
                {
                    weights[i]=reader.ReadSingle();
                    if (float.IsNaN(weights[i]) || float.IsInfinity(weights[i]))
                        throw new InvalidDataException("Non-finite actor weight");
                }
                _w[layer]=weights;
            }
            if(reader.BaseStream.Position!=reader.BaseStream.Length) throw new InvalidDataException("Trailing actor bytes");
            for (int i=0;i<_pulseAt.Length;i++) _pulseAt[i]=-100;
            for (int i=0;i<_holdAt.Length;i++) _holdAt[i]=-100;
        }

        internal PrimitiveInput Choose(float[] raw, float time)
        {
            if (_index==0)
                for (int i=0; i<4; i++) Array.Copy(raw,0,_history,i*Physical,Physical);
            Array.Copy(raw,_x,Physical);
            Array.Copy(_history,((_index+3)%4)*Physical,_x,Physical,Physical);
            Array.Copy(_history,(_index%4)*Physical,_x,2*Physical,Physical);
            Array.Copy(raw,0,_history,(_index%4)*Physical,Physical);
            int k=3*Physical;
            for (int j=0;j<Pulses.Length;j++) _x[k++]=Math.Min(time-_pulseAt[j],1f);
            for (int j=0;j<Holds.Length;j++)
            {
                bool held=(_previous & Holds[j])!=0;
                _x[k++]=held?1f:0f;
                _x[k++]=held?Math.Min(time-_holdAt[j],1f):0f;
            }
            Dense(_x,_w[0],_w[1],_h,true);
            Dense(_h,_w[2],_w[3],_z,true);
            Dense(_z,_w[4],_w[5],_scores,false);
            int best=0;
            for(int i=1;i<_scores.Length;i++) if (_scores[i]>_scores[best]) best=i;
            _index++;
            return _actions[best];
        }

        internal void Commit(PrimitiveInput action, float time)
        {
            uint press=action.Buttons & ~_previous;
            for(int j=0;j<Pulses.Length;j++) if ((press & Pulses[j])!=0) _pulseAt[j]=time;
            for(int j=0;j<Holds.Length;j++) if ((press & Holds[j])!=0) _holdAt[j]=time;
            _previous=action.Buttons;
        }

        private static void Dense(float[] x,float[] w,float[] bias,float[] result,bool relu)
        {
            int k=0;
            for(int j=0;j<result.Length;j++)
            {
                float sum=bias[j];
                for(int i=0;i<x.Length;i++) sum+=w[k++]*x[i];
                result[j]=relu?Math.Max(0f,sum):sum;
            }
        }
    }
}
