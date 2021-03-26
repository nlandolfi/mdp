package mdp

func (v *Vector) Add(u *Vector) *Vector {
	if len(v.Values) != len(u.Values) {
		panic("length mismatch")
	}

	w := &Vector{Values: make([]float64, len(v.Values))}
	for i := range v.Values {
		w.Values[i] = v.Values[i] + u.Values[i]
	}
	return w
}

func (v *Vector) Dot(u *Vector) float64 {
	if len(v.Values) != len(u.Values) {
		panic("length mismatch")
	}

	a := 0.0
	for i := range v.Values {
		a += v.Values[i] * u.Values[i]
	}
	return a
}

type Simulator func(s *Vector) *Vector

func (f *LinearRewardFunction) Compute(s *Vector) float64 {
	return f.Theta.Dot(s)
}
