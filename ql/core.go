package ql

import (
	"math"

	"github.com/nlandolfi/mdp"
)

type Function func(s mdp.State, a mdp.Action) float64

type Feature Function

// --- Parametrization  {{{

// Provides parameters for reinforcement learning
type Parametrizer interface {
	Alpha() float64
	Gamma() float64
	Epsilon() float64
}

// Creates a constant parametrization
func Parametrization(alpha float64, gamma float64, epsilon float64) *parameters {
	return &parameters{
		alpha:   alpha,
		gamma:   gamma,
		epsilon: epsilon,
	}
}

type parameters struct {
	alpha   float64
	gamma   float64
	epsilon float64
}

func (p *parameters) Alpha() float64 {
	return p.alpha
}

func (p *parameters) Gamma() float64 {
	return p.gamma
}

func (p *parameters) Epsilon() float64 {
	return p.epsilon
}

// --- }}}

type Process interface {
	// Inherited:
	//	Actions(s mdp.State) []mdp.Action
	//	Reward(s mdp.State, a mdp.Action, sPrime mdp.State) float64
	mdp.Process

	// Alpha, Gamma
	Parameters() Parametrizer

	// Features
	Features() []Feature

	// Weights
	Weights() []float64
}

func Q(p Process) Function {
	return func(s mdp.State, a mdp.Action) float64 {
		feats := p.Features()
		weights := p.Weights()

		sum := 0.0

		for i, f := range feats {
			sum += f(s, a) * weights[i]
		}

		return sum
	}
}

func max(actions []mdp.Action, s mdp.State, q Function) float64 {
	max := math.Inf(-1)

	for _, a := range actions {
		t := q(s, a)
		if t > max {
			max = t
		}
	}

	return max
}

func Update(p Process, e mdp.Episode) []float64 {
	weights := p.Weights()
	features := p.Features()
	q := Q(p)

	difference := e.Reward() + p.Parameters().Gamma()*max(p.Actions(e.NextState()), e.NextState(), q) - q(e.State(), e.Action())

	newWeights := make([]float64, len(weights))

	for i, w := range weights {
		newWeights[i] = w + p.Parameters().Alpha()*(difference)*features[i](e.State(), e.Action())
	}

	return newWeights
}

type Simulation struct {
	state mdp.State
	Process
	mdp.Policy
	Simulate func(Process, mdp.State, mdp.Action) mdp.Episode
	Update   func([]float64)
}

func (sim *Simulation) Start(start mdp.State, terminal func(s mdp.State) bool, stateDump chan<- mdp.State) mdp.State {
	states := make(chan mdp.State)
	actions := make(chan mdp.Action)
	episodes := make(chan mdp.Episode)
	weights := make(chan []float64)

	go func() {
		count := 0
		for s := range states {
			if sim.state != nil {
				stateDump <- sim.state
			}
			sim.state = s
			if terminal(s) {
				break
			}
			actions <- sim.Policy(sim.Process, s)
			a := sim.Process.Parameters().(*parameters).alpha
			sim.Process.Parameters().(*parameters).alpha = math.Max(0.05, a-1e-4)
			e := sim.Process.Parameters().(*parameters).epsilon
			sim.Process.Parameters().(*parameters).epsilon = math.Max(0, e-1e-4)
			count += 1
		}

		close(actions)
		close(stateDump)
	}()

	go func() {
		for a := range actions {
			episodes <- sim.Simulate(sim.Process, sim.state, a)
		}
		close(episodes)
	}()

	go func() {
		for e := range episodes {
			weights <- Update(sim.Process, e)
			states <- e.NextState()
		}
		close(weights)
		close(states)
	}()

	states <- start

	for w := range weights {
		sim.Update(w)
	}

	return sim.state
}
