package main

// UnionFind è una struttura dati per gestire l'unione e la ricerca delle etichette
type UnionFind struct {
	parent []int
	rank   []int
}

func NewUnionFind(size int) *UnionFind {
	parent := make([]int, size)
	rank := make([]int, size)
	for i := range parent {
		parent[i] = i
	}
	return &UnionFind{parent, rank}
}

func (uf *UnionFind) Find(x int) int {
	if uf.parent[x] != x {
		uf.parent[x] = uf.Find(uf.parent[x])
	}
	return uf.parent[x]
}

func (uf *UnionFind) Union(x, y int) {
	rootX := uf.Find(x)
	rootY := uf.Find(y)
	if rootX != rootY {
		if uf.rank[rootX] > uf.rank[rootY] {
			uf.parent[rootY] = rootX
		} else if uf.rank[rootX] < uf.rank[rootY] {
			uf.parent[rootX] = rootY
		} else {
			uf.parent[rootY] = rootX
			uf.rank[rootX]++
		}
	}
}