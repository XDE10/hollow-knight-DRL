using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using HKRLEnvMod.Observation;

namespace HKRLEnvMod.Action
{
    // Explicitly enabled on dedicated game copies. Only writes ordinary controller inputs.
    internal sealed class SistersTactics : IDisposable
    {
        private const int Frames = 20;
        private const float Dt = .05f;
        private const int Drop = 289970528, Fall = 291868896, Land = -1523407744;
        private const int Setup = -1376807296, Wind = -737267072, Charge = -1374908928, Rest = 1928179328;
        private const int WallSet = 907425216, WallWind = -1403342080, WallThrow = 739257600;
        private const int WallWait = -1368652544, WallEnd = 1871902336;
        private readonly string _file;
        private readonly SistersPolicyHead _head = new();
        private float _headWeight;
        private bool _physicsModel, _sequence, _groundTakeoff;
        private float _hurtAt;
        private int _previousHp;
        private readonly Dictionary<string,float> _param = new();
        private readonly Dictionary<int,(int state,float since)> _states = new();
        private readonly Dictionary<int,int> _discSide = new();
        private readonly Dictionary<int,(float time,float vx,float ax)> _discMotion = new();
        private readonly Box[,] _future = new Box[Frames,20];
        private readonly int[] _count = new int[Frames];
        private ulong _episode, _lastTick, _lastCommit;
        private readonly Dictionary<int,int> _evadeSide = new();
        private float _jumpEnd, _jumpAt, _lastDash, _lastAttack, _lastCast, _healAt, _healHp;
        private int _lastMove, _castAim;
        private float _riskWeight, _attackWeight, _home, _lag, _margin, _hold, _dropGap, _discBias;
        private float _castRisk, _focusHp, _hpPriority, _artAt, _artReleased;
        private bool _heal, _dash, _cast, _smart, _arts;
        public PrimitiveInput Input { get; private set; }

        private struct Box
        {
            internal float x,y,w,h,attackValue;
            internal bool disc;
        }

        private SistersTactics(string file) { _file=file; }
        public static SistersTactics? FromEnvironment()
        {
            var file=Environment.GetEnvironmentVariable("HKRL_TACTICS_POLICY");
            if(string.IsNullOrWhiteSpace(file))return null;
            if(Environment.GetEnvironmentVariable("HKRL_COMPACT_RENDER")=="1")
            {
                UnityEngine.QualitySettings.vSyncCount=0;
                UnityEngine.Application.targetFrameRate=120;
            }
            return new SistersTactics(file!);
        }
        public bool NeedsObservation(ulong tick, ulong episode, ulong commits)
        {
            if(episode==_episode && (tick-_lastTick<2 || commits==_lastCommit))return false;
            _lastCommit=commits;
            return true;
        }
        private float Param(string k,float fallback) => _param.TryGetValue(k,out var v) ? v : fallback;
        private static float Clip(float x,float lo,float hi) => Math.Min(hi,Math.Max(lo,x));
        private static float Abs(float x) => Math.Abs(x);
        private static int Sign(float x) => x>0 ? 1 : x<0 ? -1 : 0;
        private static bool IsWall(int s) => s==WallSet || s==WallWind || s==WallThrow || s==WallWait || s==WallEnd;
        private static bool IsTwinWall(int s) => s==735357952 || s==157285984 || s==7946978 || s==1756965248;

        private void Reset(ulong id)
        {
            _param.Clear();
            foreach(var raw in File.ReadAllLines(_file))
            {
                var pair=raw.Split('=');
                if(pair.Length==2) _param[pair[0].Trim()]=float.Parse(pair[1].Trim(),CultureInfo.InvariantCulture);
            }
            _riskWeight=Param("risk",65);_attackWeight=Param("offense",1.7f);_home=Param("home",28);
            _lag=Param("lag",.07f);_margin=Param("margin",.28f);_hold=Param("hold",.27f);
            _heal=Param("heal",1)>0;_dash=Param("dash",1)>0;_cast=Param("cast",1)>0;
            _smart=Param("smart",0)>0;_arts=Param("arts",0)>0;
            _headWeight=Param("head_weight",0);
            _physicsModel=Param("physics",0)>0;
            _sequence=Param("sequence",0)>0;_groundTakeoff=false;
            _hurtAt=-10;_previousHp=-1;
            _head.Reset(_file,id,_headWeight!=0);
            _dropGap=Param("drop_gap",4.9f);_discBias=Param("disc_bias",.08f);
            _castRisk=Param("cast_risk",.29f);_focusHp=Param("focus_hp",5);_hpPriority=Param("hp_priority",.6f);
            _discMotion.Clear();_artAt=-10;_artReleased=-10;
            _states.Clear();_discSide.Clear();_jumpEnd=_jumpAt=_lastDash=_lastAttack=_lastCast=_healAt=-10;
            _evadeSide.Clear();
            _lastMove=0;_episode=id; Input=PrimitiveInput.Noop;
            File.WriteAllText(_file+".active",id.ToString(CultureInfo.InvariantCulture)+"\n"+File.ReadAllText(_file));
        }

        private void Add(int i,float x,float y,float w,float h,bool disc,float value=1)
        {
            int n=_count[i];
            if(n>=20 || x<8 || x>52 || y<2 || y>24) return;
            _future[i,n]=new Box{x=x,y=y,w=w,h=h,disc=disc,attackValue=value};
            _count[i]++;
        }

        private void Predict(ObservationSnapshot obs,float now)
        {
            Array.Clear(_count,0,Frames);
            foreach(var e in obs.Entities)
            {
                bool disc=e.EntityType==HKRL.EntityType.Projectile && e.Team==HKRL.Team.Enemy;
                if(e.EntityType!=HKRL.EntityType.Boss && !disc) continue;
                int state=unchecked((int)(float)e.FsmStateHash);
                if(!_states.TryGetValue(e.EntityId,out var prev) || prev.state!=state)
                {
                    _states[e.EntityId]=prev=(state,now);
                    if(state==Drop || (state==Fall && !_evadeSide.ContainsKey(e.EntityId)))
                    {
                        float delta=obs.Player.PosX-e.PosX;
                        int side=Abs(delta)>.8f?Sign(delta):(obs.Player.PosX<30?1:-1);
                        if(_sequence)side=SelectDropSide(obs,e.EntityId,e.PosX,side);
                        if(e.PosX+side*4.6f<21.8f || e.PosX+side*4.6f>38.2f)side=-side;
                        _evadeSide[e.EntityId]=side;
                    }
                }
                float age=now-prev.since, x=e.HurtboxCenterX,y=e.HurtboxCenterY;
                float w=e.HurtboxSizeX*.5f,h=e.HurtboxSizeY*.5f;
                if(disc)
                {
                    if(w<=0 || h<=0 || x<5 || x>55)continue;
                    if(!_discSide.TryGetValue(e.EntityId,out int acc))
                    {
                        float d=e.VelX*e.VelX/48;
                        acc=Abs(x-d-18)<Abs(x+d-42)?1:-1; _discSide[e.EntityId]=acc;
                    }
                    float acceleration=acc*24;
                    if(_smart)
                    {
                        // One of the twin discs can turn around at the centre of the arena.
                        // Estimate its acceleration from observed changes in velocity.
                        if(!_discMotion.TryGetValue(e.EntityId,out var motion))
                            motion=(now,e.VelX,(x<30?-1:1)*(Abs(e.VelX)<21?12:24));
                        float dt=now-motion.time;
                        if(dt>=.055f && dt<=.5f)
                        {
                            float measured=(e.VelX-motion.vx)/dt;
                            if(Abs(measured)>4 && Abs(measured)<41)
                                motion=(now,e.VelX,Sign(measured)*(Abs(measured)<18?12:24));
                        }
                        if(dt>.5f)motion=(now,e.VelX,motion.ax);
                        _discMotion[e.EntityId]=motion;acceleration=motion.ax;
                    }
                    for(int i=0;i<Frames;i++){
                        float t=(i+1)*Dt;
                        Add(i,x+e.VelX*t+acceleration*.5f*t*t,y+e.VelY*t,w,h,true);
                    }
                    continue;
                }
                bool dropping=state==Drop || state==Fall || state==Land;
                bool charging=state==Setup || state==Wind || state==Charge || state==Rest;
                if(!dropping && !charging && !IsWall(state) && (w<=0 || y>17))continue;
                for(int i=0;i<Frames;i++)
                {
                    float t=(i+1)*Dt, bx=x,by=y,bw=w,bh=h;
                    if(dropping)
                    {
                        float until=state==Drop? Math.Max(0,.51f-age):0;
                        if(state!=Land && t>=until)
                        {
                            float travel=t-until;
                            float sx=state==Fall? e.PosX : x;
                            float shift=state==Fall?e.VelX:7*Sign(obs.Player.PosX-sx);
                            bx=sx+shift*Math.Min(.125f,travel);
                            float from=state==Fall?y:17;
                            by=Math.Max(7.3f,from-65*travel);
                            bw=.72f;bh=by>8?2.5f:1.2f;
                            if(_physicsModel)
                            {
                                float height=state==Fall?y:e.PosY-1.36f;
                                float direction=state==Fall?Sign(e.VelX):Sign(obs.Player.PosX-e.PosX);
                                float falling=Math.Max(0,(height-7.3f)/65);
                                bx=e.PosX+direction*7*Math.Min(falling,travel);
                                if(travel<falling)bx-=direction*.46f;
                                by=Math.Max(7.28f,height-65*travel);
                            }
                        }
                        if(state==Land && age+t>.64f)continue;
                    }
                    else if(charging)
                    {
                        int direction=state==Charge?Sign(e.VelX):e.PosX<30?1:-1;
                        float end=direction>0?37.6f:22.3f;
                        float until=state==Setup?Math.Max(0,.26f-age)+.40f:state==Wind?Math.Max(0,.40f-age):0;
                        if(state!=Rest && t>=until)
                        {
                            float pos=e.PosX+direction*55*(t-until);
                            bool passed=(pos-end)*direction>=0;
                            bx=passed?end:pos;by=passed?6.65f:7.48f;bw=passed?1.65f:2.65f;bh=passed?.94f:.76f;
                        }
                        if(state==Rest && age+t>.62f)continue;
                    }
                    else if(IsWall(state) || (_smart && IsTwinWall(state)))
                    {
                        // Infer the disc before its collider is visible. Its measured acceleration is 24 units/s².
                        if(state==WallSet || state==WallWind || state==WallThrow || (_smart && (state==157285984 || state==7946978)))
                        {
                            float release=state==WallSet||state==157285984?.82f-age:state==WallWind||state==7946978?.4f-age:-age;
                            float flight=t-release;
                            if(flight>=0 && flight<(_smart && release<-.08f?.04f:.85f))
                            {
                                int d=e.PosX<30?1:-1;
                                bool low=e.PosY<10.65f;
                                float height=IsTwinWall(state)?e.PosY-1.85f:low?7.15f:10.05f;
                                float vy=IsTwinWall(state)?-2.2f:low?1.1f:-1.1f;
                                Add(i,(d>0?20.0f:40.0f)+d*(34.5f*flight-12*flight*flight),
                                    height+vy*flight,2.41f,.67f,true);
                            }
                        }
                    }
                    float value=_physicsModel && e.MaxHp>0?1+_hpPriority*(1-(float)e.Hp/e.MaxHp):1;
                    if(bw>0 && bh>0)Add(i,bx,by,bw,bh,false,value);
                }
            }
        }

        private int SelectDropSide(ObservationSnapshot obs,int id,float x,int fallback)
        {
            var p=obs.Player;
            float best=float.NegativeInfinity;
            int selected=fallback;
            for(int side=-1;side<=1;side+=2)
            {
                float goal=x+side*_dropGap;
                float value=.20f*Clip((p.PosX-x)*side,-3,3)
                    +.72f*Clip(p.VelX*side/8.3f,-1,1)
                    +(_lastMove==side?.55f:0);
                value-=2.8f*Math.Max(0,Math.Max(22-goal,goal-38));
                foreach(var other in obs.Entities)
                {
                    if(other.EntityId==id || other.EntityType!=HKRL.EntityType.Boss || other.HurtboxSizeX<=0)continue;
                    int s=unchecked((int)(float)other.FsmStateHash);
                    if(s!=Land && s!=Rest && s!=Fall && s!=Drop)continue;
                    float ahead=(other.PosX-p.PosX)*side;
                    if(ahead>-.35f && ahead<4.8f)value-=1.9f*(1-Math.Max(0,ahead-2.4f)/3);
                    if(ahead<-.35f && ahead> -3.7f)value+=.85f;
                }
                if(value>best){best=value;selected=side;}
            }
            return selected;
        }

        // Integrate held lift, ballistic motion and early jump release in game seconds.
        // The second jump has a measured windup before upward velocity is applied.
        private static float Flight(float y,float v,float seconds,float lift,float release)
        {
            const float gravity=47.4f, ground=7.41f;
            float powered=Math.Min(seconds,Math.Min(lift,release));
            y+=powered*v;
            float ballistic=Math.Min(seconds,release)-powered;
            y+=ballistic*(v-.5f*gravity*ballistic);v-=gravity*ballistic;
            float after=Math.Max(0,seconds-release);
            y+=after*(Math.Min(0,v)-.5f*gravity*after);
            return Math.Max(ground,y);
        }

        private float ForecastHeight(PlayerObservation p,float now,float t,float delay,float hold,bool plan)
        {
            float existingLift=p.VelY>14.8f?Math.Max(0,.22f-(now-_jumpAt)):0;
            float existingRelease=p.VelY>0 && now<_jumpEnd+.035f?Math.Max(0,_jumpEnd+.025f-now):2;
            float currentTime=plan?Math.Min(t,delay):t;
            float y=p.OnGround?7.41f:Flight(p.PosY,p.VelY,currentTime,existingLift,existingRelease);
            if(!plan || t<=delay)return y;
            float windup=p.OnGround?0:.085f;
            float elapsed=t-delay;
            if(windup>0)
            {
                float pending=Math.Min(windup,elapsed);
                y=Flight(y,Math.Min(p.VelY,0),pending,0,2);
                if(elapsed<=windup)return y;
            }
            return Flight(y,p.OnGround?15.7f:16.8f,elapsed-windup,.205f,Math.Max(0,hold-windup));
        }

        private float PathCost(PlayerObservation p,float now,int move,float moveTime,float jumpDelay,float hold,bool dash,
                               out float immediateRisk,int followMove=0)
        {
            float cost=0,worst=0, closeRisk=0;
            bool jumpPlanned=jumpDelay<1;
            float delay=_lag+jumpDelay, dx0=p.VelX*_lag*.65f;
            float takeoffHeight=p.PosY;
            for(int i=0;i<Frames;i++)
            {
                float t=(i+1)*Dt, active=Math.Max(0,t-_lag);
                float x=p.PosX+p.VelX*Math.Min(t,_lag)*.65f+move*8.3f*Math.Min(active,moveTime);
                x+=followMove*8.3f*Math.Max(0,active-moveTime);
                if(dash)x+=move*17.5f*Math.Min(active,.22f);
                float air=t, y=p.PosY, v=p.VelY;
                if(p.OnGround) {y=7.41f;v=0;air=0;}
                if(jumpPlanned && t>=delay)
                {
                    takeoffHeight=p.OnGround?7.41f:Math.Max(7.41f,p.PosY+p.VelY*delay-23.7f*delay*delay);
                    air=t-delay; y=takeoffHeight; v=15.7f;
                    float lift=Math.Min(air,Math.Min(.20f,hold));
                    y+=v*lift;air-=lift;
                    if(t-delay>hold && hold<.27f){float rest=Math.Max(0,hold-lift); y+=v*rest-23.7f*rest*rest;v=0;air=t-delay-hold;}
                }
                else if(!p.OnGround && p.VelY>0 && now<_jumpEnd)
                {
                    float lift=Math.Min(t,Math.Min(.16f,_jumpEnd-now));
                    if(p.VelY>14){y+=p.VelY*lift;air-=lift;}
                }
                y=Math.Max(7.41f,y+v*air-23.7f*air*air);
                if(_physicsModel)y=ForecastHeight(p,now,t,delay,hold,jumpPlanned);
                float risk=0,opportunity=0,bestGap=13;
                for(int n=0;n<_count[i];n++)
                {
                    var b=_future[i,n]; float dx=Abs(x-b.x),dy=Abs(y-b.y);
                    float gx=dx-b.w-.52f, gy=dy-b.h-(_sequence?1.12f:.85f);
                    float gap=Math.Max(gx,gy);
                    float r=Clip((.16f+_margin+(_smart && b.disc?_discBias:0)-gap)/(.50f+_margin),0,1.7f);
                    if(!b.disc)
                    {
                        float hx=Clip((3.65f+b.w-dx)/1.3f,0,1)*Clip((1.1f+b.h-dy)/.65f,0,1);
                        float vertical=Clip((1.25f+b.w-dx)/.8f,0,1)*Clip((4.15f+b.h-dy)/1.5f,0,1);
                        opportunity=Math.Max(opportunity,b.attackValue*Math.Max(hx,vertical));
                        bestGap=Math.Min(bestGap,Math.Max(0,dx-b.w-2.3f)+.5f*Math.Max(0,dy-b.h-2.4f));
                    }
                    risk=Math.Max(risk,r);
                }
                if(dash && active>.025f && active<.18f)risk*=.08f;
                if(_physicsModel && now+t-_hurtAt<.95f)risk*=.24f;
                float spike=Math.Max(21.15f-x,x-38.85f);
                if(spike>-.5f && y<8.6f)risk=Math.Max(risk,Clip((spike+.5f)*2.5f,0,3));
                float weight=1-.53f*t;
                worst=Math.Max(worst,weight*risk);
                if(t<=.35f)closeRisk=Math.Max(closeRisk,risk);
                cost+=weight*(risk*_riskWeight*.15f - _attackWeight*opportunity + .10f*bestGap+.033f*Abs(x-_home));
            }
            immediateRisk=closeRisk;
            cost+=_riskWeight*worst;
            if(jumpPlanned)cost+=1.6f*(1-jumpDelay*.6f);
            if(dash)cost+=5;
            if(move!=0)cost+=.23f;
            if(move!=_lastMove)cost+=.22f;
            return cost;
        }

        public void Decide(ObservationSnapshot obs,ulong tick)
        {
            if(obs.Global.EpisodeId!=_episode)Reset(obs.Global.EpisodeId);
            _lastTick=tick;
            float now=obs.Global.TimeInEpisode;
            var p=obs.Player;
            if(_previousHp>p.Hp)_hurtAt=now;
            _previousHp=p.Hp;
            _head.Observe(obs);
            Predict(obs,now);
            float best=float.MaxValue, chosenRisk=0;
            int chosenMove=0;float chosenDelay=2,chosenHold=0;bool chosenDash=false;
            bool canJump=p.OnGround || (p.DoubleJumpAvailable && p.VelY<2 && now-_jumpAt>.32f);
            bool releaseMissedTakeoff=_sequence && _groundTakeoff && p.OnGround && now-_jumpAt>.10f && now<_jumpEnd;
            if(!p.OnGround)_groundTakeoff=false;
            if(releaseMissedTakeoff){_jumpEnd=now-.06f;_groundTakeoff=false;}
            if(now < _jumpEnd+.04f)canJump=false;
            if(releaseMissedTakeoff)canJump=false;
            if(Param("mode",0)>0)
            {
                float target=_home, targetValue=-100,lead=Param("lead",.31f), reach=Param("reach",2.9f);
                float soon=1.5f;
                foreach(var b in obs.Entities)
                {
                    if(b.EntityType!=HKRL.EntityType.Boss)continue;
                    int s=unchecked((int)(float)b.FsmStateHash);
                    float bx=b.HurtboxCenterX,dx=bx-p.PosX,age=now-_states[b.EntityId].since;
                    if(s==Wind || s==Setup || s==Charge)
                    {
                        int dir=s==Charge?Sign(b.VelX):(b.PosX<30?1:-1);
                        float ahead=(p.PosX-bx)*dir;
                        if(ahead> -1)soon=Math.Min(soon,(s==Wind?Math.Max(0,.42f-age):s==Setup?Math.Max(0,.28f-age)+.42f:0)
                            +Math.Max(0,ahead-2.5f)/55f);
                    }
                    if(s!=Land && s!=Rest && !IsWall(s) && s!=Wind && s!=Setup)continue;
                    float value=10-Abs(dx)*.30f;
                    if(s==Land || s==Rest)value+=2;
                    if(value>targetValue)
                    {
                        targetValue=value;
                        target=bx-(Abs(dx)>.5f?Sign(dx):Sign(bx-30))*reach;
                        if(IsWall(s))target=bx<30?21.9f:38.1f;
                        if(s==Wind||s==Setup)target=Clip(target,23.6f,36.4f);
                    }
                }
                float err=Clip(target,21.95f,38.05f)-p.PosX;
                chosenMove=Abs(err)>Math.Max(0,Sign(err)*p.VelX)*_lag+.38f?Sign(err):0;
                for(int i=0;i<Frames;i++)
                {
                    float t=(i+1)*Dt;float px=p.PosX+p.VelX*Math.Min(t,_lag)+chosenMove*8.3f*Math.Max(0,Math.Min(.32f,t-_lag));
                    for(int n=0;n<_count[i];n++)
                    {
                        var b=_future[i,n];
                        if(b.disc && Abs(px-b.x)<b.w+.82f && Abs(p.PosY-b.y)<b.h+.97f)soon=Math.Min(soon,t);
                    }
                }
                if(canJump && soon<lead){chosenDelay=0;chosenHold=_hold;}
                chosenRisk=Clip((.38f-soon)*3,0,1);
                if(_dash && p.CanDash && now-_lastDash>1.55f && Abs(err)>8 && soon>.45f &&
                    p.PosX+Sign(err)*5.8f>21.8f && p.PosX+Sign(err)*5.8f<38.2f)chosenDash=true;
            }
            else foreach(int m in Moves)
            foreach(float stop in Stops)
            {
                if(m==0 && stop!=Stops[0])continue;
                foreach(float jd in Jumps)
                {
                    if(jd<1 && !canJump)continue;
                    foreach(float h in Holds)
                    {
                        if(jd>=1 && h!=Holds[0])continue;
                        float hold=h<.2f?h:_hold;
                        float cost=PathCost(p,now,m,stop,jd,hold,false,out float risk);
                        if(_sequence && m!=0 && stop<.5f)
                        {
                            float reverse=PathCost(p,now,m,stop,jd,hold,false,out float reverseRisk,-m)+1.4f;
                            if(reverse<cost){cost=reverse;risk=reverseRisk;}
                        }
                        cost-=_headWeight*_head.Preference(m,jd==0 || now<_jumpEnd,false);
                        if(cost<best) {best=cost;chosenMove=m;chosenDelay=jd;chosenHold=hold;chosenDash=false;chosenRisk=risk;}
                    }
                }
                if(_dash && m!=0 && p.CanDash && now-_lastDash>1.60f)
                {
                    float cost=PathCost(p,now,m,stop,2,0,true,out float risk);
                    cost-=_headWeight*_head.Preference(m,now<_jumpEnd,true);
                    if(cost<best) {best=cost;chosenMove=m;chosenDelay=2;chosenDash=true;chosenRisk=risk;}
                }
            }
            uint buttons=0;int aim=0;
            if(chosenDelay==0){_jumpAt=now;_jumpEnd=now+chosenHold;_groundTakeoff=p.OnGround;buttons|=1;}
            if(now<_jumpEnd)buttons|=2;
            if(chosenDash){_lastDash=now;buttons|=4;}
            if(now-_lastDash<.075f)buttons|=4;
            float targetScore=-1,darkScore=0,artScore=-100;int facing=-Sign(p.Facing),artFacing=0;
            int wantedFacing=0,wantedAim=0, potentialCast=9; bool safeHeal=false;
            int live=0, dropMove=0;float nearest=100,dropUrgency=0, stationarySafe=1.0f;
            for(int i=0;i<Frames;i++)
            {
                for(int n=0;n<_count[i];n++)
                {
                    var box=_future[i,n];
                    if(Abs(p.PosX-box.x)<box.w+.94f && Abs(7.41f-box.y)<box.h+1.16f)
                        stationarySafe=Math.Min(stationarySafe,(i+1)*Dt);
                }
            }
            foreach(var b in obs.Entities)
            {
                if(b.EntityType!=HKRL.EntityType.Boss || b.HurtboxSizeX<=0 || b.HurtboxCenterY>21)continue;
                live++;
                float dx=b.HurtboxCenterX-p.PosX,dy=b.HurtboxCenterY-p.PosY;
                float w=b.HurtboxSizeX*.5f,h=b.HurtboxSizeY*.5f;
                int s=unchecked((int)(float)b.FsmStateHash);
                if((s==Drop || s==Fall) && _evadeSide.TryGetValue(b.EntityId,out int side))
                {
                    float separation=(p.PosX-b.PosX)*side;
                    float urgency=_dropGap-separation;
                    if(urgency>dropUrgency)
                    {dropUrgency=urgency;dropMove=side;}
                }
                float ax=Abs(dx),ay=Abs(dy);
                nearest=Math.Min(nearest,ax+.6f*ay);
                bool horiz=ax<w+3.85f && ay<h+1.20f;
                bool vertical=ax<w+1.35f && ay<h+4.15f && (dy>0 || !p.OnGround);
                if(_arts && ax<w+5.8f && ay<h+2.15f && dx*facing>-.7f)
                {
                    float priority=8-ax*.4f-ay*.4f;
                    if(priority>artScore){artScore=priority;artFacing=Sign(dx);}
                }
                if(horiz || vertical)
                {
                    float quality=10-ax*.35f-ay*.2f+(vertical?1:0)
                        +(_smart&&b.MaxHp>0?_hpPriority*(1-(float)b.Hp/b.MaxHp):0);
                    if(quality>targetScore){targetScore=quality;wantedFacing=Sign(dx);wantedAim=vertical && (ay>1.1f || !horiz)?Sign(dy):0;}
                }
                if(IsWall(s) && dy>1.0f && ax<3.6f)potentialCast=1;
                if(IsWall(s) && facing==Sign(dx) && ay<2.25f+h && ax>4 && potentialCast==9)potentialCast=0;
                if(ax<5 && (ay<3 || ((s==Fall || s==Drop) && ay<9)))darkScore+=1;
                if((s==WallSet && dy>2.5f && ax>9) || ((s==Land||s==Rest) && ax>3.4f && _states[b.EntityId].since>now-.16f))safeHeal=true;
            }
            if(_healAt>=0 && (p.Hp!=_healHp || now-_healAt>1.40f || stationarySafe<Math.Max(.18f,1.12f-(now-_healAt)) || dropUrgency>0))_healAt=-10;
            if(_heal && p.Hp<p.MaxHp && p.Soul>=33 && p.CanFocus && p.OnGround && now-_lastCast>.7f && _healAt<0 &&
               stationarySafe>=.99f && dropUrgency<=0 && (safeHeal || (live==0 && nearest>15)))
            {_healAt=now;_healHp=p.Hp;}
            if(_healAt>=0)
            {Input=new PrimitiveInput(0,0,32);_lastMove=0;return;}
            if(targetScore>=0 && wantedAim==0 && facing!=wantedFacing && wantedFacing!=0 && chosenRisk<.4f &&
               (wantedFacing>0?p.PosX<38.35f:p.PosX>21.65f))chosenMove=wantedFacing;
            if(targetScore>=0 && p.CanAttack && now-_lastAttack>.25f)
            {_lastAttack=now;aim=wantedAim;}
            if(now-_lastAttack<.072f){buttons|=8;aim=wantedAim;}
            if(_cast && p.Soul>=33 && p.CanCast && now-_lastCast>.50f && !chosenDash)
            {
                if(darkScore>=1 && (chosenRisk>(_smart?_castRisk:.25f) || darkScore>=2 || !p.OnGround))potentialCast=-1;
                if(_smart && stationarySafe<.26f && darkScore>=1 && (p.Soul>=55 || p.Hp<=3))potentialCast=-1;
                bool escape=_sequence && stationarySafe<=.30f && chosenRisk>=.42f && (p.Hp<=4 || p.Soul>=66 || live>=2);
                if(escape)potentialCast=-1;
                if(_sequence && !escape && potentialCast!=-1 && p.Hp<7 && p.Soul<66)potentialCast=9;
                if(potentialCast!=9 && (escape || p.Hp>=(_smart?_focusHp:4) || p.Soul>=55 || darkScore>=2 || (_smart && potentialCast==-1 && stationarySafe<.26f)))
                {_castAim=potentialCast;_lastCast=now;}
            }
            if(now-_lastCast<.080f){buttons=(buttons & ~8u)|16;aim=_castAim;}
            if(dropUrgency>0 && Param("drop_guard",1)>0)
            {
                bool follow=true;
                if(_sequence && live>=2 && chosenMove!=dropMove)
                {
                    PathCost(p,now,dropMove,.48f,chosenDelay,chosenHold,chosenDash,out float alternativeRisk);
                    follow=alternativeRisk<chosenRisk+.26f;
                }
                if(follow)chosenMove=dropMove;
                if(_sequence && dropUrgency>1.5f && aim==0 && facing==chosenMove)
                    buttons&=~8u;
                if(p.OnGround && stationarySafe>.12f && (!_smart || (chosenRisk<.24f && live<=1))) {buttons &= ~3u;_jumpEnd=-10;}
                if(dropUrgency>2.5f && stationarySafe<.24f && _dash && p.CanDash && now-_lastDash>1.55f)
                {buttons|=4;_lastDash=now;}
            }
            if(p.PosX<21.4f && chosenMove<0 || p.PosX>38.6f && chosenMove>0)chosenMove=0;
            if(_arts && (buttons&16)==0 && now-_artReleased>.12f)
            {
                bool charging=_artAt>=0;
                if(charging && (now-_artAt>=1.4f || p.NailChargeTimer>=1.35f) && artScore>0)
                {
                    buttons=(buttons&~8u)|256u;aim=0;
                    if(artFacing!=0 && artFacing!=facing && chosenRisk<.24f)chosenMove=artFacing;
                    _artReleased=now;_artAt=-10;_lastAttack=now;
                }
                else if(charging && now-_artAt>3 && targetScore>=0 && artScore<=0)
                {_artReleased=now;_artAt=-10;buttons&=~8u;}
                else if(charging || (p.CanNailCharge && targetScore<0 && nearest>8 && now-_lastAttack>.35f))
                {if(!charging)_artAt=now;buttons|=128u;}
            }
            _lastMove=chosenMove;Input=new PrimitiveInput(chosenMove,aim,buttons);
        }
        private static readonly int[] Moves={0,-1,1};
        private static readonly float[] Stops={.20f,.48f,1};
        private static readonly float[] Jumps={2,0,.15f,.30f,.50f};
        private static readonly float[] Holds={.15f,.30f};
        public void Record(float time) => _head.Record(time,Input);
        public void Flush() => _head.Flush();
        public void Dispose() => _head.Dispose();
    }
}
