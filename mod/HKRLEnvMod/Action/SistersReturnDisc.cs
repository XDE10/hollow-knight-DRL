using System;

namespace HKRLEnvMod.Action
{
    internal static class SistersReturnDisc
    {
        // Relative motion between planning samples and the next accepted input
        // can hide a narrow overlap. Only widen the fast, low return arc.
        internal static float Sweep(float verticalSpeed, float horizontalSpeed,
                                    float acceleration, float centerHeight)
        {
            if (verticalSpeed >= -1.6f || horizontalSpeed * acceleration <= 0f ||
                centerHeight > 12.3f)
                return 0f;
            return Math.Min(1.45f, .15f + .045f * Math.Abs(horizontalSpeed));
        }
    }
}
