package mdp

// A State is a model of the world
type State interface{}

// An Action is a choice which can be taken
type Action interface{}

// A Process is the interface for an MDP, at the highest level
type Process interface {
	// Possible Actions
	Actions(s State) []Action

	// Reward function of the MDP.
	Reward(s State, a Action, sPrime State) float64
}

// A KnownProcess is a process for which the states
// and transition function is known
type KnownProcess interface {
	Process

	// Possible States
	States() []State

	// Outcomes of being in state s and taking action a
	Outcomes(s State, a Action) []State

	// Transition function of the MDP.
	// P(s' | s, a)
	Transition(s State, a Action, sPrime State) float64
}

// An Episode is an experience
type Episode interface {
	// Starting state
	State() State

	// What action was taken
	Action() Action

	// Immediate reward
	Reward() float64

	// Where you ended up
	NextState() State
}

// A policy maps states to actions
type Policy func(p Process, s State) Action
