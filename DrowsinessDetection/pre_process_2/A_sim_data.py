import pandas as pd

"""
Dados temporais de uma trip do condutor no simulador

Simuladores:

    Simulator 1
        40 trips
        4 trips de cada condutor
        Condutores com id impar

    Simulador 2
        40 trips
        4 trips por condutor
        Condutores com ID par

Features:
    timer [s]
    ove_body_velocity
    ove_odometer
    OpenDrivePosition   Road  Pos                         Yaw  D  ################# Verificar
    OVE road
    OVE s
    OVE r
    OVE vel [m/s]
    ove_road_velocity
    ove_body_acceleration
    ove_road_acceleration
    ove_body_jerk
    ove_road_jerk
    throttle
    brake_pedal_active
    brake_force
    steering_wheel_angle
    left_indicator
    right_indicator
    highbeam
    lowbeam
    showing_pdt
    log_pdt_dir
    pdt_state
    pdt_reaction_time
    kss_answer
    kss_question_visible
    ahead_distance
    behind_distance
    left_line_crossing
    right_line_crossing
    vitaport_value
"""

# load the CSV file
file_path = '../datasets_2/valu3s/sim/simulator_1_elvagar/valu3s_db_fpfp01_2_night_1678126655829.csv'
data = pd.read_csv(file_path, delimiter=';')

# number of instances (rows)
num_instances = data.shape[0]

# number of features (columns)
num_features = data.shape[1]

# feature names
feature_names = data.columns.tolist()

# basic statistics for numerical features
basic_stats = data.describe()

# output the results
print("\nNumber of instances:", num_instances)
print("Number of features:", num_features)
print("Feature names:", feature_names)
print("\nBasic statistics:")
print(basic_stats)


