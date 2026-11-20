using System;
using System.Globalization;
using System.IO;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Action
{
    /// <summary>Optional learned movement preferences for the Sisters trajectory planner.</summary>
    internal sealed class SistersPolicyHead : IDisposable
    {
        public const int Features = 56;
        public const int Hidden = 32;
        public const int Actions = 12;
        private readonly float[] _features = new float[Features];
        private readonly float[] _hidden = new float[Hidden];
        private readonly float[] _q = new float[Actions];
        private readonly float[] _bossDistances = new float[3];
        private readonly float[] _discDistances = new float[2];
        private float[]? _weights;
        private SistersWorldStudent? _world;
        private float _baseScale, _worldScale;
        private string? _usageFile;
        private ulong _usageEpisode;
        private int _decisions, _different, _baselineChoice;
        private float _bestBaseline;
        private BinaryWriter? _trace;
        private int _rows;
        private static float Clip(float v) => Math.Max(-2f, Math.Min(2f,v));

        public void Reset(string config, ulong episode, float baseScale, float worldScale)
        {
            Dispose();
            _weights = null;
            _baseScale = baseScale;
            _worldScale = worldScale;
            _world = worldScale == 0 ? null : new SistersWorldStudent(config + ".wm.weights");
            _usageFile = _world == null ? null : config + ".usage";
            _usageEpisode = episode;
            _decisions = _different = 0;
            Array.Clear(_q,0,Actions);
            if (baseScale != 0)
            {
                var path=config+".weights";
                var values=File.ReadAllLines(path);
                if(values.Length!=Hidden*Features+Hidden+Actions*Hidden+Actions)
                    throw new InvalidDataException("Unexpected Sisters policy weight count.");
                _weights=Array.ConvertAll(values,v=>float.Parse(v,CultureInfo.InvariantCulture));
            }
            if(Environment.GetEnvironmentVariable("HKRL_TACTICS_TRACE")!="1")return;
            _trace=new BinaryWriter(new FileStream(config+".trace",FileMode.Create,FileAccess.Write,FileShare.ReadWrite));
            _trace.Write(0x31534B48);
            _trace.Write(episode);
            _trace.Write(Features);
            _rows=0;
        }

        // Stable physical slots sorted by distance; no episode, instance ID or seed enters the policy.
        public void Observe(ObservationSnapshot obs)
        {
            if(_trace==null && _weights==null && _world==null)return;
            WriteFeatures(obs, _features, _bossDistances, _discDistances);
            ScoreFeatures();
        }

        internal static void WriteFeatures(ObservationSnapshot obs, float[] _features, float[] _bossDistances, float[] _discDistances)
        {
            var p=obs.Player;
            Array.Clear(_features,0,Features);
            _features[0]=Clip((p.PosX-30)/10); _features[1]=Clip((p.PosY-7.41f)/8);
            _features[2]=Clip(p.VelX/15); _features[3]=Clip(p.VelY/20);
            _features[4]=p.Hp/(float)Math.Max(1,p.MaxHp); _features[5]=p.Soul/99f;
            _features[6]=p.OnGround?1:0; _features[7]=p.DoubleJumpAvailable?1:0;
            _features[8]=p.CanDash?1:0; _features[9]=p.Invulnerable?1:0;
            _features[10]=-p.Facing; _features[11]=p.CanAttack?1:0;
            for(int i=0;i<3;i++)_bossDistances[i]=float.MaxValue;
            for(int i=0;i<2;i++)_discDistances[i]=float.MaxValue;
            foreach(var e in obs.Entities)
            {
                bool boss=e.EntityType==HKRL.EntityType.Boss;
                if(!boss && !(e.EntityType==HKRL.EntityType.Projectile && e.Team==HKRL.Team.Enemy))continue;
                float x=e.HurtboxCenterX,y=e.HurtboxCenterY;
                if(e.HurtboxSizeX<=0 || y>25 || x<10 || x>50)continue;
                float dx=x-p.PosX,dy=y-p.PosY;
                float distance=Math.Abs(dx)+.65f*Math.Abs(dy);
                var distances=boss?_bossDistances:_discDistances;
                int start=boss?12:42, width=boss?10:7;
                int slot=0;
                while(slot<distances.Length&&distance>=distances[slot])slot++;
                if(slot==distances.Length)continue;
                for(int j=distances.Length-1;j>slot;j--)
                {
                    distances[j]=distances[j-1];
                    Array.Copy(_features,start+(j-1)*width,_features,start+j*width,width);
                }
                distances[slot]=distance;
                int n=start+slot*width;
                _features[n]=1; _features[n+1]=Clip(dx/12); _features[n+2]=Clip(dy/9);
                _features[n+3]=Clip(e.VelX/(boss?55:32)); _features[n+4]=Clip(e.VelY/(boss?60:3));
                if(boss)
                {
                    _features[n+5]=e.Hp/(float)Math.Max(e.MaxHp,1);
                    int state=unchecked((int)(float)e.FsmStateHash);
                    _features[n+6]=state==289970528||state==291868896?1:0;
                    _features[n+7]=state== -1376807296||state== -737267072||state== -1374908928?1:0;
                    _features[n+8]=state== -1523407744||state==1928179328?1:0;
                    _features[n+9]=x<20||x>40?1:0;
                }
                else
                {
                    _features[n+5]=Clip(e.HurtboxSizeX/5);
                    _features[n+6]=Clip(e.HurtboxSizeY/2);
                }
            }
        }

        private void ScoreFeatures()
        {
            _world?.Observe(_features);
            if(_weights==null)return;
            int k=0;
            for(int j=0;j<Hidden;j++)
            {
                float sum=_weights[Hidden*Features+j];
                for(int i=0;i<Features;i++)sum+=_weights[k++]*_features[i];
                _hidden[j]=Math.Max(0,sum);
            }
            k+=Hidden;
            for(int j=0;j<Actions;j++)
            {
                float sum=_weights[_weights.Length-Actions+j];
                for(int i=0;i<Hidden;i++)sum+=_weights[k++]*_hidden[i];
                _q[j]=sum;
            }
        }

        public float Preference(int move,bool jump,bool dash)
        {
            int index=(move+1)*4+(jump?2:0)+(dash?1:0);
            return _baseScale*_q[index]+_worldScale*(_world?.Value(index) ?? 0);
        }

        public void BeginChoice() => _bestBaseline = float.PositiveInfinity;

        public void Compare(float cost, int move, bool jump, bool dash)
        {
            if (_world == null) return;
            int code = (move + 1) * 4 + (jump ? 2 : 0) + (dash ? 1 : 0);
            float baseline = cost + _worldScale * _world.Value(code);
            if (baseline < _bestBaseline) { _bestBaseline = baseline; _baselineChoice = code; }
        }

        public void FinalizeChoice(int move, bool jump, bool dash)
        {
            if (_world == null || float.IsPositiveInfinity(_bestBaseline)) return;
            _decisions++;
            int code = (move + 1) * 4 + (jump ? 2 : 0) + (dash ? 1 : 0);
            if (_baselineChoice != code) _different++;
        }

        public void Record(float time,PrimitiveInput input)
        {
            if(_trace==null)return;
            _trace.Write(time);
            foreach(float x in _features)_trace.Write(x);
            _trace.Write((float)input.MovementX);_trace.Write((float)input.AimY);_trace.Write((float)input.Buttons);
            if(++_rows%32==0)_trace.Flush();
        }

        public void Flush()
        {
            _trace?.Flush();
            if (_usageFile != null)
                File.WriteAllText(_usageFile, string.Join(" ", _usageEpisode, _decisions, _different));
        }
        public void Dispose() { _trace?.Dispose();_trace=null; }
    }
}
