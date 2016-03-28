package mdp

import "math"

// --- {{{

type Values interface {
	For(s State) float64

	Update(s State, value float64)
}

func ValueIteration(p KnownProcess, depth int, v Values, discount float64) {
	if depth == 0 {
		return
	}

	for _, s := range p.States() {
		v.Update(s, valueIterationUpdate(s, p, v, discount))
	}

	ValueIteration(p, depth-1, v, discount)
}

func valueIterationUpdate(s State, p KnownProcess, v Values, discount float64) float64 {
	max := math.Inf(-1)
	for _, a := range p.Actions(s) {
		sum := 0.0
		for _, sPrime := range p.Outcomes(s, a) {
			sum += p.Transition(s, a, sPrime) * (p.Reward(s, a, sPrime) + discount*v.For(sPrime))
		}
		if sum > max {
			max = sum
		}
	}
	return max
}

// --- }}}

// --- Q-Value Iteration {{{

type QValues interface {
	For(s State, a Action) float64

	Update(s State, a Action, value float64)
}

func QValueIteration(p KnownProcess, depth int, qv QValues, discount float64) {
	if depth == 0 {
		return
	}

	for _, s := range p.States() {
		for _, a := range p.Actions(s) {
			qv.Update(s, a, qValueIterationUpdate(s, a, p, qv, discount))
		}
	}

	QValueIteration(p, depth, qv, discount)
}

func qValueIterationUpdate(s State, a Action, p KnownProcess, qv QValues, discount float64) float64 {
	sum := 0.0
	for _, sPrime := range p.Outcomes(s, a) {
		sum += p.Transition(s, a, sPrime) * (p.Reward(s, a, sPrime) + discount*maxQValue(sPrime, p, qv))
	}
	return sum
}

func maxQValue(s State, p KnownProcess, qv QValues) float64 {
	max := math.Inf(-1)
	for _, a := range p.Actions(s) {
		if v := qv.For(s, a); v > max {
			max = v
		}
	}
	return max
}

// --- }}}
