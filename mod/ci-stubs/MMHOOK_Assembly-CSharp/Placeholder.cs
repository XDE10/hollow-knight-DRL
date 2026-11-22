namespace On
{
    public static class GameManager
    {
        public delegate void orig_SetTimeScale_float(
            global::GameManager self,
            float newTimeScale);

        public delegate void hook_SetTimeScale_float(
            orig_SetTimeScale_float orig,
            global::GameManager self,
            float newTimeScale);

        public static event hook_SetTimeScale_float? SetTimeScale_float;

        public static void RaiseSetTimeScaleForTests(
            global::GameManager self,
            float newTimeScale)
        {
            void Original(global::GameManager _, float scale)
            {
                global::TimeController.GenericTimeScale = scale;
            }

            hook_SetTimeScale_float? hook = SetTimeScale_float;
            if (hook == null)
            {
                Original(self, newTimeScale);
                return;
            }

            hook(Original, self, newTimeScale);
        }
    }
}

namespace On
{
    public static class HeroController
    {
        public delegate void orig_TakeDamage(global::HeroController self, UnityEngine.GameObject hit,
            GlobalEnums.CollisionSide side, int amount, int hazardType);
        public delegate void hook_TakeDamage(orig_TakeDamage orig, global::HeroController self,
            UnityEngine.GameObject hit, GlobalEnums.CollisionSide side, int amount, int hazardType);
        public static event hook_TakeDamage? TakeDamage { add { } remove { } }
    }
}

namespace On
{
    public static class HealthManager
    {
        public delegate void orig_Hit(global::HealthManager self, global::HitInstance hit);
        public delegate void hook_Hit(orig_Hit orig, global::HealthManager self, global::HitInstance hit);
        public static event hook_Hit? Hit { add { } remove { } }
    }
}
