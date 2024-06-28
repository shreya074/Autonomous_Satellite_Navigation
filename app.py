import numpy as np
from flask import Flask, render_template, request
from tensorflow.keras import layers, models # type: ignore

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('Sattelite DRL.html')

@app.route('/next_move', methods=['POST'])
def next_move():
    proximity = request.form['proximity']
    movement_front_back = int(request.form['movement_front'])
    movement_front_back = int(request.form['movement_back'])
    movement_right_left = int(request.form['movement_right'])
    movement_right_left = int(request.form['movement_left'])


    try:
        next_move = determine_next_move(proximity, movement_front_back, movement_right_left)
    except Exception as e:
        print("An error occurred during reinforcement learning:", e)
        next_move = "unknown"

    next_move = next_move.replace('up', 'front').replace('down', 'back')

    return render_template('NextMove.html', next_move=next_move)

def build_ddpg_actor(input_shape, num_actions):
    inputs = layers.Input(shape=input_shape)
    x = layers.Dense(64, activation='relu')(inputs)
    x = layers.Dense(64, activation='relu')(x)
    outputs = layers.Dense(num_actions, activation='tanh')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

def build_ddpg_critic(input_shape, num_actions):
    state_input = layers.Input(shape=input_shape)
    action_input = layers.Input(shape=(num_actions,))
    
    
    state_pathway = layers.Dense(32, activation='relu')(state_input)
    state_pathway = layers.Dense(64, activation='relu')(state_pathway)
    
    
    action_pathway = layers.Dense(64, activation='relu')(action_input)
    
    
    combined = layers.Add()([state_pathway, action_pathway])
    outputs = layers.Dense(1)(combined) 
    model = models.Model(inputs=[state_input, action_input], outputs=outputs)
    return model

input_shape = (3,) 
num_actions = 4    
ddpg_actor = build_ddpg_actor(input_shape, num_actions)
ddpg_critic = build_ddpg_critic(input_shape, num_actions)

try:
    ddpg_actor.load_weights('ddpg_actor_weights.h5')
    ddpg_critic.load_weights('ddpg_critic_weights.h5')
except Exception as e:
    print("An error occurred while loading model weights:", e)


def determine_next_move(proximity, movement_front_back, movement_right_left):
    
    proximity_binary = 1 if proximity == 'near' else 0
    
    state = np.array([[proximity_binary, movement_front_back, movement_right_left]])

    action = ddpg_actor.predict(state)[0]

    if proximity == 'far':
        next_move = "Go on in the same move"
    elif proximity == 'near':
        if movement_front_back != 0 and movement_right_left != 0:
            next_move = determine_combined_move(movement_front_back, movement_right_left)
        elif movement_front_back != 0:
            next_move = determine_single_move(movement_front_back, 'front', 'back')
        elif movement_right_left != 0:
            next_move = determine_single_move(movement_right_left, 'right', 'left')
        else:
            # Otherwise, select the move based on the dominant action
            if np.abs(action[0]) > np.abs(action[1]):
                next_move = 'front' if action[0] > 0 else 'back'
            else:
                next_move = 'left' if action[1] > 0 else 'right'
    else:
        # If there is no object near, continue in the same move
        next_move = "Go on in the same move"

    return next_move

def determine_single_move(movement, positive_move, negative_move):
    return positive_move if movement > 0 else negative_move

def determine_combined_move(front_back_movement, right_left_movement):
    if front_back_movement > 0:
        return 'front' if right_left_movement > 0 else 'right'
    else:
        return 'back' if right_left_movement > 0 else 'left'

if __name__ == "__main__":
    app.run(debug=True)
