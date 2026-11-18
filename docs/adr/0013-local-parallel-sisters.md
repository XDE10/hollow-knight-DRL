# ADR-0013. Share a recurrent policy across local game instances

## Decision

Each game instance owns a loopback port, save-slot copy, episode state and GRU state.
The demonstration collector writes a disjoint set of numbered episodes per worker.
Synchronous local PPO gives each worker an equal share of one update's decision budget.
The learner waits for all workers, preserves independently bootstrapped advantages, joins
buffers across the environment dimension and updates one set of weights. Games that finish
sampling early pause until the next update. The existing single-game path remains available.

For a single evaluation task, assign whole episodes to workers by global episode index modulo
the number of games. Reset each episode's GRU state and retain the original seed for that
index. Compare frozen candidates and controls on the same port order and game count.

## Consequences

Real games consume enough CPU and memory to change how much game time passes between
decisions. Compare total decisions per wall second and physics ticks per decision when
choosing how many instances to run. Keep each port on loopback and preserve personal saves
before using game slots for training copies. See the [slot-3 measurements](../sisters_survival.md).
