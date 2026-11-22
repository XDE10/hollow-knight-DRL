namespace HKRLEnvMod.Action
{
    // The mixed recorder observes both proposals but sends just one to the game.
    internal struct SistersDemonstrationDash
    {
        private const float OnsetWindow = .12001f;
        private bool _held;
        private float? _pendingLearnerPress;

        internal float Observe(float previous, float now, bool dashing)
        {
            if (_pendingLearnerPress is not float pressedAt)
                return previous;

            if (now - pressedAt > OnsetWindow)
                _pendingLearnerPress = null;
            else if (dashing)
            {
                _pendingLearnerPress = null;
                return pressedAt > previous ? pressedAt : previous;
            }

            return previous;
        }

        internal float Commit(float previous, float planned, float now,
            bool canStartDash, PrimitiveInput executed, bool teacherActs)
        {
            bool pressed = (executed.Buttons & (1u << ActionMasker.ButtonDash)) != 0;
            bool rising = pressed && !_held;
            _held = pressed;

            if (rising)
                _pendingLearnerPress = !teacherActs && canStartDash ? now : null;

            return teacherActs ? planned : previous;
        }
    }
}
