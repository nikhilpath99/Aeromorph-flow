from aeromorph_flow.src.geometry.morphing import morph_airfoils
from aeromorph_flow.src.geometry.naca import generate_naca4


def test_naca4_shapes():
    airfoil = generate_naca4("2412", n_points=64)
    assert airfoil["x"].shape == (64,)
    assert airfoil["yu"].shape == (64,)
    assert airfoil["yl"].shape == (64,)
    assert airfoil["coords"].shape[1] == 2


def test_morph_path_length():
    a = generate_naca4("0012", n_points=32)
    b = generate_naca4("4415", n_points=32)
    path = morph_airfoils(a, b, n_steps=4)
    assert len(path) == 4

