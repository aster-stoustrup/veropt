from veropt.interfaces.experiment_utility import ExperimentalState, Point

point_0 = Point(
    parameters = {'c_k' : 0.1, 'c_eps' : 0.01},
    objective_value = -10.0
)

point_1 = Point(
    parameters = {'c_k' : 0.02, 'c_eps' : 0.1},
    objective_value = -20.0
)

state = ExperimentalState(
    experiment_name = "test_experiment",
    experiment_directory = ""
)

new_points = {
    0 : point_0,
    1 : point_1
}

print(state.model_dump())

point_no = len(new_points)
start = state.next_point
end = state.next_point + point_no

for i in range(start, end):
    state.update(
        new_point = new_points[i],
    )

print(state.model_dump())