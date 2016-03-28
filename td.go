package mdp

func TemporalDifferenceLearning(samples <-chan Episode, p Process, alpha, gamma float64, v Values) {
	for e := range samples {
		TemporalDifferenceUpdate(p, e, alpha, gamma, v)
	}
}

func TemporalDifferenceUpdate(p Process, e Episode, alpha, gamma float64, v Values) {
	v.Update(e.State(), (1-alpha)*v.For(e.State())+alpha*(e.Reward()+gamma*v.For(e.NextState())))
}
