import sys, random, enum, ast, time, csv
import numpy as np
from matrx import grid_world
from brains1.ArtificialBrain import ArtificialBrain
from actions1.CustomActions import *
from matrx import utils
from matrx.grid_world import GridWorld
from matrx.agents.agent_utils.state import State
from matrx.agents.agent_utils.navigator import Navigator
from matrx.agents.agent_utils.state_tracker import StateTracker
from matrx.actions.door_actions import OpenDoorAction
from matrx.actions.object_actions import GrabObject, DropObject, RemoveObject
from matrx.actions.move_actions import MoveNorth
from matrx.messages.message import Message
from matrx.messages.message_manager import MessageManager
from actions1.CustomActions import RemoveObjectTogether, CarryObjectTogether, DropObjectTogether, CarryObject, Drop


class Phase(enum.Enum):
    INTRO = 1,
    FIND_NEXT_GOAL = 2,
    PICK_UNSEARCHED_ROOM = 3,
    PLAN_PATH_TO_ROOM = 4,
    FOLLOW_PATH_TO_ROOM = 5,
    PLAN_ROOM_SEARCH_PATH = 6,
    FOLLOW_ROOM_SEARCH_PATH = 7,
    PLAN_PATH_TO_VICTIM = 8,
    FOLLOW_PATH_TO_VICTIM = 9,
    TAKE_VICTIM = 10,
    PLAN_PATH_TO_DROPPOINT = 11,
    FOLLOW_PATH_TO_DROPPOINT = 12,
    DROP_VICTIM = 13,
    WAIT_FOR_HUMAN = 14,
    WAIT_AT_ZONE = 15,
    FIX_ORDER_GRAB = 16,
    FIX_ORDER_DROP = 17,
    REMOVE_OBSTACLE_IF_NEEDED = 18,
    ENTER_ROOM = 19


class BaselineAgent(ArtificialBrain):
    def __init__(self, slowdown, condition, name, folder):
        super().__init__(slowdown, condition, name, folder)
        # Initialization of some relevant variables
        self._tick = None
        self._slowdown = slowdown
        self._condition = condition
        self._human_name = name
        self._folder = folder
        self._phase = Phase.INTRO
        self._room_vics = []
        self._searched_rooms = []
        self._found_victims = []
        self._collected_victims = []
        self._found_victim_logs = {}
        self._send_messages = []
        self._current_door = None
        self._team_members = []
        self._carrying_together = False
        self._remove = False
        self._goal_vic = None
        self._goal_loc = None
        self._human_loc = None
        self._distance_human = None
        self._distance_drop = None
        self._agent_loc = None
        self._todo = []
        self._answered = False
        self._to_search = []
        self._carrying = False
        self._waiting = False
        self._rescue = None
        self._recent_vic = None
        self._received_messages = []
        self._moving = False
        self.WEIGHT_RESCUED_SUCCESSFULLY = 0.15
        self.WEIGHT_REMOVING_OBJECT_TOGETHER = 0.1
        self.WEIGHT_HUMAN_CALLS_OBSTACLE_BUT_LEAVES = 0.07
        self.WEIGHT_HUMAN_CORRECTLY_COMMUNICATES_VICTIM_LOCATION = 0.1
        self.WEIGHT_HUMAN_INCORRECTLY_COMMUNICATES_VICTIM_LOCATION = 0.15
        self.WEIGHT_HUMAN_RESCUES_VICTIM = 0.1
        self.WEIGHT_HUMAN_LIED_ABOUT_RESCUE_VICTIM = 0.17
        self.WEIGHT_HUMAN_SEARCHES_AREA = 0.05
        self.WEIGHT_HUMAN_LIED_ABOUT_SEARCHING_AREA = 0.1
        self.WEIGHT_HUMAN_DOES_NOT_COME_RESCUE_MILD_VICTIM = 0.08
        self.WEIGHT_HUMAN_COMES_RESCUE_MILD_VICTIM = 0.05
        self.WEIGHT_HUMAN_DOES_NOT_COME_RESCUE_CRITICAL_VICTIM = 0.2
        self.WEIGHT_HUMAN_COMES_RESCUE_CRITICAL_VICTIM = 0.12
        self.WEIGHT_HUMAN_FINDS_VICTIM = 0.08
        self._max_waiting_ticks = 300           # 100 for debugging. Change to 300
        self._ticks_since_waiting = None
        self._object_to_remove = None
        self._object_to_remove_id = None
        self._rooms_searched_by_human = []
        self._rooms_searched_by_agent = []
        self._victims_rescued_by_human = []
        self._victims_rescued_by_agent = []
        self._interactions = None

    def initialize(self):
        # Initialization of the state tracker and navigation algorithm
        self._state_tracker = StateTracker(agent_id=self.agent_id)
        self._navigator = Navigator(agent_id=self.agent_id, action_set=self.action_set,
                                    algorithm=Navigator.A_STAR_ALGORITHM)

    def filter_observations(self, state):
        # Filtering of the world state before deciding on an action 
        return state

    def compute_trust(self, trust, alpha):
        """
        Maps a trust value in [-1,1] to [0,1] using a logistic function.
        The confidence alpha in [0,1] is scaled to [0,5] to set the steepness.

        :param trust:  float in [-1, 1], the current trust level
        :param alpha:  float in [0, 1], the confidence level
        :return:       float in [0, 1], the logistic-mapped trust
        """
        # Scale alpha from [0,1] to [0,5]
        alpha_scaled = 5.0 * alpha

        return 1.0 / (1.0 + np.exp(-alpha_scaled * trust))

    def map_interactions_to_confidence(self, gamma=30):
        """
            Map the number of interactions to [0, 1] using a negative exponential to represent confidence.
        """
        return 1.0 - np.exp(-self._interactions / gamma)

    def _decide_with_trust(self, task):
        """
            Applies a logistic transform to the trust values to better represent trust, and makes a final decision based on that value.

            ALWAYS-TRUST: Return True
            NEVER-TRUST: Return False
            RANDOM-TRUST: Choose random number in [0, 1], if > 0.5 return True
        """

        confidence = self.map_interactions_to_confidence()
        trustBeliefs = self._loadCurrentBeliefs(self._team_members, self._folder)
        trust_value = (trustBeliefs[self._human_name][task]['willingness'] + trustBeliefs[self._human_name][task]['competence']) / 2
        final_trust = self.compute_trust(trust_value, confidence)

        # Generate random number in [0, 1] to make a final decision
        number = np.random.uniform(0, 1)
        if number <= final_trust:
            return True
        return False

    def decide_on_actions(self, state):
        # Identify team members
        agent_name = state[self.agent_id]['obj_id']
        for member in state['World']['team_members']:
            if member != agent_name and member not in self._team_members:
                self._team_members.append(member)
        # Create a list of received messages from the human team member
        for mssg in self.received_messages:
            for member in self._team_members:
                if mssg.from_id == member and mssg.content not in self._received_messages:
                    self._received_messages.append(mssg.content)
        # Process messages from team members
        self._process_messages(state, self._team_members, self._condition)
        # Initialize and update trust beliefs for team members
        trustBeliefs = self._loadCurrentBeliefs(self._team_members, self._folder)

        # Check whether human is close in distance
        if state[{'is_human_agent': True}]:
            self._distance_human = 'close'
        if not state[{'is_human_agent': True}]:
            # Define distance between human and agent based on last known area locations
            if self._agent_loc in [1, 2, 3, 4, 5, 6, 7] and self._human_loc in [8, 9, 10, 11, 12, 13, 14]:
                self._distance_human = 'far'
            if self._agent_loc in [1, 2, 3, 4, 5, 6, 7] and self._human_loc in [1, 2, 3, 4, 5, 6, 7]:
                self._distance_human = 'close'
            if self._agent_loc in [8, 9, 10, 11, 12, 13, 14] and self._human_loc in [1, 2, 3, 4, 5, 6, 7]:
                self._distance_human = 'far'
            if self._agent_loc in [8, 9, 10, 11, 12, 13, 14] and self._human_loc in [8, 9, 10, 11, 12, 13, 14]:
                self._distance_human = 'close'

        # Define distance to drop zone based on last known area location
        if self._agent_loc in [1, 2, 5, 6, 8, 9, 11, 12]:
            self._distance_drop = 'far'
        if self._agent_loc in [3, 4, 7, 10, 13, 14]:
            self._distance_drop = 'close'

        # Check whether victims are currently being carried together by human and agent 
        for info in state.values():
            if 'is_human_agent' in info and self._human_name in info['name'] and len(
                    info['is_carrying']) > 0 and 'critical' in info['is_carrying'][0]['obj_id'] or \
                    'is_human_agent' in info and self._human_name in info['name'] and len(
                info['is_carrying']) > 0 and 'mild' in info['is_carrying'][0][
                'obj_id'] and self._rescue == 'together' and not self._moving:
                # If victim is being carried, add to collected victims memory
                if info['is_carrying'][0]['img_name'][8:-4] not in self._collected_victims:
                    self._collected_victims.append(info['is_carrying'][0]['img_name'][8:-4])
                self._carrying_together = True
            if 'is_human_agent' in info and self._human_name in info['name'] and len(info['is_carrying']) == 0:
                self._carrying_together = False
        # If carrying a victim together, let agent be idle (because joint actions are essentially carried out by the human)
        if self._carrying_together == True:
            return None, {}

        # Send the hidden score message for displaying and logging the score during the task, DO NOT REMOVE THIS
        self._send_message('Our score is ' + str(state['rescuebot']['score']) + '.', 'RescueBot')

        # Ongoing loop until the task is terminated, using different phases for defining the agent's behavior
        while True:
            if Phase.INTRO == self._phase:
                # Send introduction message
                self._send_message('Hello! My name is RescueBot. Together we will collaborate and try to search and rescue the 8 victims on our right as quickly as possible. \
                Each critical victim (critically injured girl/critically injured elderly woman/critically injured man/critically injured dog) adds 6 points to our score, \
                each mild victim (mildly injured boy/mildly injured elderly man/mildly injured woman/mildly injured cat) 3 points. \
                If you are ready to begin our mission, you can simply start moving.', 'RescueBot')
                # Initialize trust beliefs
                trustBeliefs = self._loadBelief(self._team_members, self._folder)
                self._trustBelief(self._team_members, trustBeliefs, self._folder, self._received_messages)
                # Wait untill the human starts moving before going to the next phase, otherwise remain idle
                if not state[{'is_human_agent': True}]:
                    self._phase = Phase.FIND_NEXT_GOAL
                else:
                    return None, {}

            if Phase.FIND_NEXT_GOAL == self._phase:
                # Definition of some relevant variables
                self._answered = False
                self._goal_vic = None
                self._goal_loc = None
                self._rescue = None
                self._moving = True
                remaining_zones = []
                remaining_vics = []
                remaining = {}
                # Identification of the location of the drop zones
                zones = self._get_drop_zones(state)
                # Identification of which victims still need to be rescued and on which location they should be dropped
                for info in zones:
                    if str(info['img_name'])[8:-4] not in self._collected_victims:
                        remaining_zones.append(info)
                        remaining_vics.append(str(info['img_name'])[8:-4])
                        remaining[str(info['img_name'])[8:-4]] = info['location']
                if remaining_zones:
                    self._remainingZones = remaining_zones
                    self._remaining = remaining
                # Remain idle if there are no victims left to rescue
                if not remaining_zones:
                    # Save current beliefs to allTrustBeliefs.csv
                    self._saveTrustBeliefs(self._team_members, self._folder)
                    return None, {}

                # Check which victims can be rescued next because human or agent already found them
                for vic in remaining_vics:
                    # Define a previously found victim as target victim because all areas have been searched
                    if vic in self._found_victims and vic in self._todo and len(self._searched_rooms) == 0:
                        self._goal_vic = vic
                        self._goal_loc = remaining[vic]

                        # If robot trusts human and the victim condition is mild, or the condition is critical, the robot asks for human help
                        if ('mild' in self._goal_vic and self._decide_with_trust("rescue")) or 'critical' in self._goal_vic:
                            # Move to target victim
                            self._rescue = 'together'
                            self._send_message('Moving to ' + self._found_victim_logs[vic][
                                'room'] + ' to pick up ' + self._goal_vic + '. Please come there as well to help me carry ' + self._goal_vic + ' to the drop zone.',
                                              'RescueBot')
                        # Otherwise rescue the victim alone if human is not trustworthy
                        else:
                            self._rescue = 'alone'
                            self._send_message('Moving to ' + self._found_victim_logs[vic][
                                'room'] + ' to pick up ' + self._goal_vic + '.','RescueBot')
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                    # Define a previously found victim as target victim
                    if vic in self._found_victims and vic not in self._todo:
                        self._goal_vic = vic
                        self._goal_loc = remaining[vic]
                        # Rescue together when victim is critical or when the human is weak and the victim is mildly injured
                        if 'critical' in vic or 'mild' in vic and self._condition == 'weak':
                            self._rescue = 'together'
                            # Rescue alone if the robot does not trust the human and the victim condition is mild
                            if 'mild' in vic and not self._decide_with_trust("rescue"):
                                self._rescue = 'alone'
                        # Rescue alone if the victim is mildly injured and the human not weak
                        if 'mild' in vic and self._condition != 'weak':
                            self._rescue = 'alone'
                        # Plan path to victim because the exact location is known (i.e., the agent found this victim)
                        if 'location' in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_VICTIM
                            return Idle.__name__, {'duration_in_ticks': 25}
                        # Plan path to area because the exact victim location is not known, only the area (i.e., human found this  victim)
                        if 'location' not in self._found_victim_logs[vic].keys():
                            self._phase = Phase.PLAN_PATH_TO_ROOM
                            return Idle.__name__, {'duration_in_ticks': 25}
                    # If there are no target victims found, visit an unsearched area to search for victims
                    if vic not in self._found_victims or vic in self._found_victims and vic in self._todo and len(
                            self._searched_rooms) > 0:
                        self._phase = Phase.PICK_UNSEARCHED_ROOM

            if Phase.PICK_UNSEARCHED_ROOM == self._phase:
                agent_location = state[self.agent_id]['location']
                # Identify which areas are not explored yet
                unsearched_rooms = [room['room_name'] for room in state.values()
                                   if 'class_inheritance' in room
                                   and 'Door' in room['class_inheritance']
                                   and room['room_name'] not in self._searched_rooms
                                   and room['room_name'] not in self._to_search]
                # If all areas have been searched but the task is not finished, start searching areas again
                if self._remainingZones and len(unsearched_rooms) == 0:
                    self._to_search = []
                    self._searched_rooms = []
                    self._send_messages = []
                    self.received_messages = []
                    self.received_messages_content = []
                    self._send_message('Going to re-search all areas.', 'RescueBot')
                    self._phase = Phase.FIND_NEXT_GOAL
                # If there are still areas to search, define which one to search next
                else:
                    # Identify the closest door when the agent did not search any areas yet
                    if self._current_door == None:
                        # Find all area entrance locations
                        self._door = state.get_room_doors(self._getClosestRoom(state, unsearched_rooms, agent_location))[
                            0]
                        self._doormat = \
                            state.get_room(self._getClosestRoom(state, unsearched_rooms, agent_location))[-1]['doormat']
                        # Workaround for one area because of some bug
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        # Plan path to area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Identify the closest door when the agent just searched another area
                    if self._current_door != None:
                        self._door = \
                            state.get_room_doors(self._getClosestRoom(state, unsearched_rooms, self._current_door))[0]
                        self._doormat = \
                            state.get_room(self._getClosestRoom(state, unsearched_rooms, self._current_door))[-1][
                                'doormat']
                        if self._door['room_name'] == 'area 1':
                            self._doormat = (3, 5)
                        self._phase = Phase.PLAN_PATH_TO_ROOM

            if Phase.PLAN_PATH_TO_ROOM == self._phase:
                # Reset the navigator for a new path planning
                self._navigator.reset_full()

                # Check if there is a goal victim, and it has been found, but its location is not known
                if self._goal_vic \
                        and self._goal_vic in self._found_victims \
                        and 'location' not in self._found_victim_logs[self._goal_vic].keys():
                    # Retrieve the victim's room location and related information
                    victim_location = self._found_victim_logs[self._goal_vic]['room']
                    self._door = state.get_room_doors(victim_location)[0]
                    self._doormat = state.get_room(victim_location)[-1]['doormat']

                    # Handle special case for 'area 1'
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)

                    # Set the door location based on the doormat
                    doorLoc = self._doormat

                # If the goal victim's location is known, plan the route to the identified area
                else:
                    if self._door['room_name'] == 'area 1':
                        self._doormat = (3, 5)
                    doorLoc = self._doormat

                # Add the door location as a waypoint for navigation
                self._navigator.add_waypoints([doorLoc])
                # Follow the route to the next area to search
                self._phase = Phase.FOLLOW_PATH_TO_ROOM

            if Phase.FOLLOW_PATH_TO_ROOM == self._phase:
                # Check if the previously identified target victim was rescued by the human
                if self._goal_vic and self._goal_vic in self._collected_victims:
                    # Reset current door and switch to finding the next goal
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Check if the human found the previously identified target victim in a different room
                if self._goal_vic \
                        and self._goal_vic in self._found_victims \
                        and self._door['room_name'] != self._found_victim_logs[self._goal_vic]['room']:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL
                # Check if the human already searched the previously identified area without finding the target victim
                if self._door['room_name'] in self._searched_rooms and self._goal_vic not in self._found_victims:
                    # If robot does not trust human, search the room anyway
                    if self._decide_with_trust("search"):
                        self._current_door = None
                        self._phase = Phase.FIND_NEXT_GOAL
                    else:
                        self._searched_rooms.remove(self._door['room_name'])
                # Move to the next area to search
                else:
                    # Update the state tracker with the current state
                    self._state_tracker.update(state)

                    # Explain why the agent is moving to the specific area, either:
                    # [-] it contains the current target victim
                    # [-] it is the closest un-searched area
                    if self._goal_vic in self._found_victims \
                            and str(self._door['room_name']) == self._found_victim_logs[self._goal_vic]['room'] \
                            and not self._remove:
                        if self._condition == 'weak':
                            self._send_message('Moving to ' + str(
                                self._door['room_name']) + ' to pick up ' + self._goal_vic + ' together with you.',
                                              'RescueBot')
                        else:
                            self._send_message(
                                'Moving to ' + str(self._door['room_name']) + ' to pick up ' + self._goal_vic + '.',
                                'RescueBot')

                    if self._goal_vic not in self._found_victims and not self._remove or not self._goal_vic and not self._remove:
                        self._send_message(
                            'Moving to ' + str(self._door['room_name']) + ' because it is the closest unsearched area.',
                            'RescueBot')

                    # Set the current door based on the current location
                    self._current_door = self._door['location']

                    # Retrieve move actions to execute
                    action = self._navigator.get_move_action(self._state_tracker)
                    # Check for obstacles blocking the path to the area and handle them if needed
                    if action is not None:
                        # Remove obstacles blocking the path to the area 
                        for info in state.values():
                            if 'class_inheritance' in info and 'ObstacleObject' in info[
                                'class_inheritance'] and 'stone' in info['obj_id'] and info['location'] not in [(9, 4),
                                                                                                                (9, 7),
                                                                                                                (9, 19),
                                                                                                                (21,
                                                                                                                 19)]:
                                self._send_message('Reaching ' + str(self._door['room_name'])
                                                   + ' will take a bit longer because I found stones blocking my path.',
                                                   'RescueBot')
                                return RemoveObject.__name__, {'object_id': info['obj_id']}
                        return action, {}
                    # Identify and remove obstacles if they are blocking the entrance of the area
                    self._phase = Phase.REMOVE_OBSTACLE_IF_NEEDED

            if Phase.REMOVE_OBSTACLE_IF_NEEDED == self._phase:
                objects = []
                agent_location = state[self.agent_id]['location']
                # Identify which obstacle is blocking the entrance
                for info in state.values():
                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'rock' in info[
                        'obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:
                            self._send_message('Found rock blocking ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + ' \n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area ', '') + ' \
                                \n clock - removal time: 5 seconds \n afstand - distance between us: ' + self._distance_human,
                                              'RescueBot')
                            self._waiting = True
                            # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Wait for the human to help removing the obstacle and remove the obstacle together
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                            # Tell the human to come over and be idle untill human arrives
                            if not state[{'is_human_agent': True}]:
                                self._send_message('Please come to ' + str(self._door['room_name']) + ' to remove rock.',
                                                  'RescueBot')
                                # Start counting the time waiting for the human
                                self._ticks_since_waiting = 1
                                # If human called, but he's not here, decrease competence
                                if self._remove:
                                    trustBeliefs[self._human_name]['remove']['competence'] -= self.WEIGHT_HUMAN_CALLS_OBSTACLE_BUT_LEAVES
                                    self._trustBelief(self._team_members, trustBeliefs, self._folder,
                                                      self._received_messages)
                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                self._send_message('Lets remove rock blocking ' + str(self._door['room_name']) + '!',
                                                  'RescueBot')
                                trustBeliefs[self._human_name]['remove'][
                                    'competence'] += self.WEIGHT_REMOVING_OBJECT_TOGETHER
                                self._trustBelief(self._team_members, trustBeliefs, self._folder,
                                                  self._received_messages)
                                self._interactions += 1
                                return None, {}
                        # Remain idle untill the human communicates what to do with the identified obstacle 
                        else:
                            if self._ticks_since_waiting:
                                # If human did not arrive increase the number of ticks for waiting for them
                                self._ticks_since_waiting += 1
                                # If robot waited above threshold, move on and find another goal
                                if self._ticks_since_waiting > self._max_waiting_ticks and not state[{'is_human_agent': True}]:
                                    self._ticks_since_waiting = None
                                    self._send_message('Leaving because you did not come remove the rock.',
                                                       'RescueBot')
                                    self._answered = False
                                    self._remove = False
                                    self._waiting = False
                                    self._phase = Phase.FIND_NEXT_GOAL # Change to ENTER_ROOM?
                                    # Add area to the to do list
                                    self._to_search.append(self._door['room_name'])
                                    trustBeliefs[self._human_name]['remove']['willingness'] -= self.WEIGHT_REMOVING_OBJECT_TOGETHER
                                    self._interactions += 1
                                    self._trustBelief(self._team_members, trustBeliefs, self._folder,
                                                      self._received_messages)
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'tree' in info[
                        'obj_id']:
                        objects.append(info)
                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:
                            self._send_message('Found tree blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove" or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + '\n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area ', '') + ' \
                                \n clock - removal time: 10 seconds', 'RescueBot')
                            self._waiting = True
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the obstacle if the human tells the agent to do so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove' or self._remove:
                            if not self._remove:
                                self._answered = True
                                self._waiting = False
                                self._send_message('Removing tree blocking ' + str(self._door['room_name']) + '.',
                                                  'RescueBot')
                            if self._remove:
                                self._send_message('Removing tree blocking ' + str(
                                    self._door['room_name']) + ' because you asked me to.', 'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False
                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remain idle untill the human communicates what to do with the identified obstacle
                        else:
                            return None, {}

                    if 'class_inheritance' in info and 'ObstacleObject' in info['class_inheritance'] and 'stone' in \
                            info['obj_id']:
                        objects.append(info)

                        # If robot is not already waiting for human, make a decision
                        if not self._waiting:
                            # If robot does not trust human, they will remove the obstacle alone
                            if not self._decide_with_trust("remove"):
                                self._answered = True
                                self._waiting = False
                                self._send_message('Removing stones blocking ' + str(
                                    self._door['room_name']) + '.',
                                                   'RescueBot')
                                self._phase = Phase.ENTER_ROOM
                                self._remove = False
                                return RemoveObject.__name__, {'object_id': info['obj_id']}

                        # Communicate which obstacle is blocking the entrance
                        if self._answered == False and not self._remove and not self._waiting:
                            self._send_message('Found stones blocking  ' + str(self._door['room_name']) + '. Please decide whether to "Remove together", "Remove alone", or "Continue" searching. \n \n \
                                Important features to consider are: \n safe - victims rescued: ' + str(
                                self._collected_victims) + ' \n explore - areas searched: area ' + str(
                                self._searched_rooms).replace('area', '') + ' \
                                \n clock - removal time together: 3 seconds \n afstand - distance between us: ' + self._distance_human + '\n clock - removal time alone: 20 seconds',
                                              'RescueBot')
                            self._waiting = True
                        # Determine the next area to explore if the human tells the agent not to remove the obstacle          
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Continue' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            # Add area to the to do list
                            self._to_search.append(self._door['room_name'])
                            self._phase = Phase.FIND_NEXT_GOAL
                        # Remove the obstacle alone if the human decides so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove alone' and not self._remove:
                            self._answered = True
                            self._waiting = False
                            self._send_message('Removing stones blocking ' + str(self._door['room_name']) + '.',
                                              'RescueBot')
                            self._phase = Phase.ENTER_ROOM
                            self._remove = False

                            return RemoveObject.__name__, {'object_id': info['obj_id']}
                        # Remove the obstacle together if the human decides so
                        if self.received_messages_content and self.received_messages_content[
                            -1] == 'Remove together' or self._remove:
                            if not self._remove:
                                self._answered = True
                            if not state[{'is_human_agent': True}]:
                                self._send_message(
                                    'Please come to ' + str(self._door['room_name']) + ' to remove stones together.',
                                    'RescueBot')
                                # Start counting the time it takes for human to come
                                self._ticks_since_waiting = 1

                                return None, {}
                            # Tell the human to remove the obstacle when he/she arrives
                            if state[{'is_human_agent': True}]:
                                trustBeliefs[self._human_name]['remove']['competence'] += self.WEIGHT_REMOVING_OBJECT_TOGETHER
                                self._trustBelief(self._team_members, trustBeliefs, self._folder,
                                                  self._received_messages)
                                self._send_message('Lets remove stones blocking ' + str(self._door['room_name']) + '!',
                                                  'RescueBot')
                                self._interactions += 1
                                return None, {}
                        # Remain idle until the human communicates what to do with the identified obstacle
                        else:
                            if self._ticks_since_waiting:
                                # If human did not arrive start incrementing the ticks
                                self._ticks_since_waiting += 1
                                # If robot waited too long, start removing the obstacle alone
                                if self._ticks_since_waiting > self._max_waiting_ticks and not state[{'is_human_agent': True}]:
                                    self._ticks_since_waiting = None
                                    self._send_message('Removing stones blocking ' + str(
                                        self._door['room_name']) + ' by myself because you did not come.',
                                                       'RescueBot')
                                    trustBeliefs[self._human_name]['remove']['willingness'] -= self.WEIGHT_REMOVING_OBJECT_TOGETHER
                                    self._interactions += 1
                                    self._trustBelief(self._team_members, trustBeliefs, self._folder, self._received_messages)
                                    self._waiting = False
                                    self._phase = Phase.ENTER_ROOM
                                    self._remove = False
                                    return RemoveObject.__name__, {'object_id': info['obj_id']}

                            return None, {}

                # If no obstacles are blocking the entrance, enter the area
                if len(objects) == 0:
                    # If we have ticks initialized but no objects remaining, this means the human has arrived in time.
                    # So increase trust
                    if self._ticks_since_waiting:
                        self._ticks_since_waiting = None
                        trustBeliefs[self._human_name]['remove']['willingness'] += self.WEIGHT_REMOVING_OBJECT_TOGETHER
                        self._trustBelief(self._team_members, trustBeliefs, self._folder, self._received_messages)
                        self._interactions += 1

                    self._answered = False
                    self._remove = False
                    self._waiting = False
                    self._phase = Phase.ENTER_ROOM
                else:
                    # Check if there are objects blocking the entrance and the human reported the room as searched
                    # This means human lied about searching the room
                    if self._door['room_name'] in self._rooms_searched_by_human:
                        trustBeliefs[self._human_name]['search']['competence'] -= self.WEIGHT_HUMAN_LIED_ABOUT_SEARCHING_AREA
                        trustBeliefs[self._human_name]['search'][
                            'willingness'] -= self.WEIGHT_HUMAN_LIED_ABOUT_SEARCHING_AREA
                        self._trustBelief(self._team_members, trustBeliefs, self._folder,
                                          self._received_messages)
                        self._interactions += 1

            if Phase.ENTER_ROOM == self._phase:
                self._answered = False

                # Check if the target victim has been rescued by the human, and switch to finding the next goal
                if self._goal_vic in self._collected_victims:
                    # Check whether the robot trusts the human. If not, search the room anyway
                    if self._decide_with_trust("rescue"):
                        self._current_door = None
                        self._phase = Phase.FIND_NEXT_GOAL
                    else:
                        self._state_tracker.update(state)

                        action = self._navigator.get_move_action(self._state_tracker)
                        # If there is a valid action, return it; otherwise, plan to search the room
                        if action is not None:
                            return action, {}
                        self._phase = Phase.PLAN_ROOM_SEARCH_PATH

                # Check if the target victim is found in a different area, and start moving there
                if self._goal_vic in self._found_victims \
                        and self._door['room_name'] != self._found_victim_logs[self._goal_vic]['room']:
                    self._current_door = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Check if area already searched without finding the target victim, and plan to search another area
                if self._door['room_name'] in self._searched_rooms and self._goal_vic not in self._found_victims:
                    # Check if robot trusts human, and if not search the room anyway
                    if self._decide_with_trust("search"):
                        self._current_door = None
                        self._phase = Phase.FIND_NEXT_GOAL
                    else:
                        self._state_tracker.update(state)

                        action = self._navigator.get_move_action(self._state_tracker)
                        # If there is a valid action, return it; otherwise, plan to search the room
                        if action is not None:
                            return action, {}
                        self._phase = Phase.PLAN_ROOM_SEARCH_PATH

                # Enter the area and plan to search it
                else:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # If there is a valid action, return it; otherwise, plan to search the room
                    if action is not None:
                        return action, {}
                    self._phase = Phase.PLAN_ROOM_SEARCH_PATH

            if Phase.PLAN_ROOM_SEARCH_PATH == self._phase:
                # Extract the numeric location from the room name and set it as the agent's location
                self._agent_loc = int(self._door['room_name'].split()[-1])

                # Store the locations of all area tiles in the current room
                room_tiles = [info['location'] for info in state.values()
                             if 'class_inheritance' in info
                             and 'AreaTile' in info['class_inheritance']
                             and 'room_name' in info
                             and info['room_name'] == self._door['room_name']]
                self._roomtiles = room_tiles

                # Make the plan for searching the area
                self._navigator.reset_full()
                self._navigator.add_waypoints(self._efficientSearch(room_tiles))

                # Initialize variables for storing room victims and switch to following the room search path
                self._room_vics = []
                self._phase = Phase.FOLLOW_ROOM_SEARCH_PATH

            if Phase.FOLLOW_ROOM_SEARCH_PATH == self._phase:
                # Search the area
                self._state_tracker.update(state)
                action = self._navigator.get_move_action(self._state_tracker)
                if action != None:
                    # Identify victims present in the area
                    for info in state.values():
                        if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance']:
                            vic = str(info['img_name'][8:-4])
                            # Remember which victim the agent found in this area
                            if vic not in self._room_vics:
                                self._room_vics.append(vic)

                            # Identify the exact location of the victim that was found by the human earlier
                            if vic in self._found_victims and 'location' not in self._found_victim_logs[vic].keys():
                                self._recent_vic = vic
                                # Add the exact victim location to the corresponding dictionary
                                self._found_victim_logs[vic] = {'location': info['location'],
                                                                'room': self._door['room_name'],
                                                                'obj_id': info['obj_id']}
                                if vic == self._goal_vic:
                                    # Communicate which victim was found
                                    self._send_message('Found ' + vic + ' in ' + self._door[
                                        'room_name'] + ' because you told me ' + vic + ' was located here.',
                                                      'RescueBot')
                                    # NEW IMPLEMENTATION
                                    # Increase trust because human correctly communicated the victim's location
                                    trustBeliefs[self._human_name]['search']['competence'] += self.WEIGHT_HUMAN_CORRECTLY_COMMUNICATES_VICTIM_LOCATION
                                    trustBeliefs[self._human_name]['search']['willingness'] += self.WEIGHT_HUMAN_CORRECTLY_COMMUNICATES_VICTIM_LOCATION
                                    self._trustBelief(self._team_members, trustBeliefs, self._folder,
                                                      self._received_messages)
                                    self._interactions += 1

                                    # Add the area to the list with searched areas
                                    if self._door['room_name'] not in self._searched_rooms:
                                        self._searched_rooms.append(self._door['room_name'])
                                    # Do not continue searching the rest of the area but start planning to rescue the victim
                                    self._phase = Phase.FIND_NEXT_GOAL

                            # If we find a victim that the human previously communicated they rescued
                            if vic in self._collected_victims:
                                self._collected_victims.remove(vic)
                                self._found_victims.remove(vic)
                                self._found_victim_logs.pop(vic)

                                # Decrease trust since human lied
                                trustBeliefs[self._human_name]['rescue']['competence'] -= self.WEIGHT_HUMAN_LIED_ABOUT_RESCUE_VICTIM
                                self._trustBelief(self._team_members, trustBeliefs, self._folder, self._received_messages)
                                self._interactions += 1

                            # Identify injured victim in the area
                            if 'healthy' not in vic and vic not in self._found_victims:
                                self._recent_vic = vic
                                # Add the victim and the location to the corresponding dictionary
                                self._found_victims.append(vic)
                                self._found_victim_logs[vic] = {'location': info['location'],
                                                                'room': self._door['room_name'],
                                                                'obj_id': info['obj_id']}
                                # Communicate which victim the agent found and ask the human whether to rescue the victim now or at a later stage
                                if 'mild' in vic and self._answered == False and not self._waiting:
                                    # Check whether robot trusts human. If not, rescue alone
                                    if self._decide_with_trust("rescue"):
                                        self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue together", "Rescue alone", or "Continue" searching. \n \n \
                                            Important features to consider are: \n safe - victims rescued: ' + str(
                                            self._collected_victims) + '\n explore - areas searched: area ' + str(
                                            self._searched_rooms).replace('area ', '') + '\n \
                                            clock - extra time when rescuing alone: 15 seconds \n afstand - distance between us: ' + self._distance_human,
                                                          'RescueBot')
                                        self._waiting = True
                                    else:
                                        self._send_message(
                                            'Picking up ' + self._recent_vic + ' in ' + self._door['room_name'] + '.',
                                            'RescueBot')
                                        self._rescue = 'alone'
                                        self._answered = True
                                        self._waiting = False
                                        self._goal_vic = self._recent_vic
                                        self._goal_loc = self._remaining[self._goal_vic]
                                        self._recent_vic = None
                                        self._phase = Phase.PLAN_PATH_TO_VICTIM

                                if 'critical' in vic and self._answered == False and not self._waiting:
                                    self._send_message('Found ' + vic + ' in ' + self._door['room_name'] + '. Please decide whether to "Rescue" or "Continue" searching. \n\n \
                                        Important features to consider are: \n explore - areas searched: area ' + str(
                                        self._searched_rooms).replace('area',
                                                                      '') + ' \n safe - victims rescued: ' + str(
                                        self._collected_victims) + '\n \
                                        afstand - distance between us: ' + self._distance_human, 'RescueBot')
                                    self._waiting = True
                                    # Execute move actions to explore the area

                                # If victim is found and the human reported the area as searched before, decrease trust
                                if self._door['room_name'] in self._rooms_searched_by_human:
                                    trustBeliefs[self._human_name]['search'][
                                        'competence'] -= self.WEIGHT_HUMAN_LIED_ABOUT_SEARCHING_AREA
                                    self._trustBelief(self._team_members, trustBeliefs, self._folder,
                                                      self._received_messages)
                                    self._interactions += 1

                    return action, {}

                # Communicate that the agent did not find the target victim in the area while the human previously communicated the victim was located here
                if self._goal_vic in self._found_victims and self._goal_vic not in self._room_vics and \
                        self._found_victim_logs[self._goal_vic]['room'] == self._door['room_name']:
                    self._send_message(self._goal_vic + ' not present in ' + str(self._door[
                                                                                    'room_name']) + ' because I searched the whole area without finding ' + self._goal_vic + '.',
                                      'RescueBot')
                    # Remove the victim location from memory
                    self._found_victim_logs.pop(self._goal_vic, None)
                    self._found_victims.remove(self._goal_vic)
                    self._room_vics = []
                    # Reset received messages (bug fix)
                    self.received_messages = []
                    self.received_messages_content = []
                    # NEW IMPLEMENTATION
                    # Decrease trust since human incorrectly communicated the victim's location
                    trustBeliefs[self._human_name]['search']['competence'] -= self.WEIGHT_HUMAN_INCORRECTLY_COMMUNICATES_VICTIM_LOCATION
                    trustBeliefs[self._human_name]['search'][
                        'willingness'] -= self.WEIGHT_HUMAN_INCORRECTLY_COMMUNICATES_VICTIM_LOCATION
                    self._trustBelief(self._team_members, trustBeliefs, self._folder, self._received_messages)
                    self._interactions += 1

                # Add the area to the list of searched areas
                if self._door['room_name'] not in self._searched_rooms:
                    self._searched_rooms.append(self._door['room_name'])
                # Add the area to the list of searched areas by agent
                if self._door['room_name'] not in self._rooms_searched_by_agent:
                    self._rooms_searched_by_agent.append(self._door['room_name'])

                # Make a plan to rescue a found critically injured victim if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue' and 'critical' in self._recent_vic:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    # Tell the human to come over and help carry the critically injured victim
                    if not state[{'is_human_agent': True}]:
                        self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(
                            self._recent_vic) + ' together.', 'RescueBot')
                    # Tell the human to carry the critically injured victim together
                    if state[{'is_human_agent': True}]:
                        self._send_message('Lets carry ' + str(
                            self._recent_vic) + ' together! Please wait until I moved on top of ' + str(
                            self._recent_vic) + '.', 'RescueBot')
                    self._goal_vic = self._recent_vic
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Make a plan to rescue a found mildly injured victim together if the human decides so
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue together' and 'mild' in self._recent_vic:
                    self._rescue = 'together'
                    self._answered = True
                    self._waiting = False
                    # Tell the human to come over and help carry the mildly injured victim
                    if not state[{'is_human_agent': True}]:
                        self._send_message('Please come to ' + str(self._door['room_name']) + ' to carry ' + str(
                            self._recent_vic) + ' together.', 'RescueBot')
                    # Tell the human to carry the mildly injured victim together
                    if state[{'is_human_agent': True}]:
                        self._send_message('Lets carry ' + str(
                            self._recent_vic) + ' together! Please wait until I moved on top of ' + str(
                            self._recent_vic) + '.', 'RescueBot')
                        self._interactions += 1
                    self._goal_vic = self._recent_vic
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Make a plan to rescue the mildly injured victim alone if the human decides so, and communicate this to the human
                if self.received_messages_content and self.received_messages_content[
                    -1] == 'Rescue alone' and 'mild' in self._recent_vic:
                    self._send_message('Picking up ' + self._recent_vic + ' in ' + self._door['room_name'] + '.',
                                      'RescueBot')
                    self._rescue = 'alone'
                    self._answered = True
                    self._waiting = False
                    self._goal_vic = self._recent_vic
                    self._goal_loc = self._remaining[self._goal_vic]
                    self._recent_vic = None
                    self._phase = Phase.PLAN_PATH_TO_VICTIM
                # Continue searching other areas if the human decides so
                if self.received_messages_content and self.received_messages_content[-1] == 'Continue':
                    self._answered = True
                    self._waiting = False
                    self._todo.append(self._recent_vic)
                    self._recent_vic = None
                    self._phase = Phase.FIND_NEXT_GOAL

                # Remain idle untill the human communicates to the agent what to do with the found victim
                if self.received_messages_content and self._waiting and self.received_messages_content[
                    -1] != 'Rescue' and self.received_messages_content[-1] != 'Continue':
                    return None, {}
                # Find the next area to search when the agent is not waiting for an answer from the human or occupied with rescuing a victim
                if not self._waiting and not self._rescue:
                    self._recent_vic = None
                    self._phase = Phase.FIND_NEXT_GOAL
                return Idle.__name__, {'duration_in_ticks': 25}

            if Phase.PLAN_PATH_TO_VICTIM == self._phase:
                # Plan the path to a found victim using its location
                self._navigator.reset_full()
                self._navigator.add_waypoints([self._found_victim_logs[self._goal_vic]['location']])
                # Follow the path to the found victim
                self._phase = Phase.FOLLOW_PATH_TO_VICTIM

            if Phase.FOLLOW_PATH_TO_VICTIM == self._phase:
                # Start searching for other victims if the human already rescued the target victim
                if self._goal_vic and self._goal_vic in self._collected_victims:
                    self._phase = Phase.FIND_NEXT_GOAL

                # Move towards the location of the found victim
                else:
                    self._state_tracker.update(state)

                    action = self._navigator.get_move_action(self._state_tracker)
                    # If there is a valid action, return it; otherwise, switch to taking the victim
                    if action is not None:
                        return action, {}
                    self._phase = Phase.TAKE_VICTIM

            if Phase.TAKE_VICTIM == self._phase:
                # Store all area tiles in a list
                room_tiles = [info['location'] for info in state.values()
                             if 'class_inheritance' in info
                             and 'AreaTile' in info['class_inheritance']
                             and 'room_name' in info
                             and info['room_name'] == self._found_victim_logs[self._goal_vic]['room']]
                self._roomtiles = room_tiles
                objects = []
                # When the victim has to be carried by human and agent together, check whether human has arrived at the victim's location
                for info in state.values():
                    # When the victim has to be carried by human and agent together, check whether human has arrived at the victim's location
                    if 'class_inheritance' in info and 'CollectableBlock' in info['class_inheritance'] and 'critical' in \
                            info['obj_id'] and info['location'] in self._roomtiles or \
                            'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'mild' in info['obj_id'] and info[
                        'location'] in self._roomtiles and self._rescue == 'together' or \
                            self._goal_vic in self._found_victims and self._goal_vic in self._todo and len(
                        self._searched_rooms) == 0 and 'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'critical' in info['obj_id'] and info['location'] in self._roomtiles or \
                            self._goal_vic in self._found_victims and self._goal_vic in self._todo and len(
                        self._searched_rooms) == 0 and 'class_inheritance' in info and 'CollectableBlock' in info[
                        'class_inheritance'] and 'mild' in info['obj_id'] and info['location'] in self._roomtiles:
                        objects.append(info)
                        # Remain idle when the human has not arrived at the location
                        if not self._human_name in info['name']:
                            # NEW
                            # Start counting the time for human to arrive
                            if not self._ticks_since_waiting:
                                self._ticks_since_waiting = 1
                            else:
                                self._ticks_since_waiting += 1

                            # If the waiting time exceeds the threshold, don't wait for human
                            if self._ticks_since_waiting > self._max_waiting_ticks and not state[{'is_human_agent': True}]:
                                self._ticks_since_waiting = None
                                self._waiting = False
                                self._interactions += 1
                                # If victim is critical, move on to another area
                                if 'critical' in info['obj_id']:
                                    self._phase = Phase.FIND_NEXT_GOAL
                                    self._send_message("I am leaving because you did not come rescue the critical victim.", "RescueBot")
                                    self._todo.append(self._goal_vic)
                                    trustBeliefs[self._human_name]['rescue']['willingness'] -= self.WEIGHT_HUMAN_DOES_NOT_COME_RESCUE_CRITICAL_VICTIM
                                    self._trustBelief(self._team_members, trustBeliefs, self._folder, self._received_messages)
                                    return Idle.__name__, {'duration_in_ticks': 25}
                                if 'mild' in info['obj_id']:
                                    self._send_message("You did not come pick up the mildly injured victim. I am taking it by myself.", "RescueBot")
                                    self._rescue = 'alone'
                                    zones = self._get_drop_zones(state)
                                    # Since now we rescue alone, we need to identify the new drop location of the victim
                                    for info in zones:
                                        if str(info['img_name'])[8:-4] in self._goal_vic:
                                            self._goal_loc = info['location']
                                            break
                                    trustBeliefs[self._human_name]['rescue'][
                                        'willingness'] -= self.WEIGHT_HUMAN_DOES_NOT_COME_RESCUE_MILD_VICTIM
                                    self._trustBelief(self._team_members, trustBeliefs, self._folder,
                                                      self._received_messages)

                            else:
                                self._waiting = True
                                self._moving = False
                                return None, {}
                        else:
                            self._ticks_since_waiting = None
                # Add the victim to the list of rescued victims when it has been picked up
                if len(objects) == 0 and 'critical' in self._goal_vic or len(
                        objects) == 0 and 'mild' in self._goal_vic and self._rescue == 'together':
                    self._interactions += 1
                    # If human arrived reset the tick, and increase trust
                    if self._ticks_since_waiting:
                        self._ticks_since_waiting = None
                        if 'critical' in self._goal_vic:
                            trustBeliefs[self._human_name]['rescue']['willingness'] += self.WEIGHT_HUMAN_COMES_RESCUE_CRITICAL_VICTIM
                        if 'mild' in self._goal_vic:
                            trustBeliefs[self._human_name]['rescue'][
                                'willingness'] += self.WEIGHT_HUMAN_COMES_RESCUE_MILD_VICTIM
                        self._trustBelief(self._team_members, trustBeliefs, self._folder, self._received_messages)
                    self._waiting = False
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                        self._victims_rescued_by_agent.append(self._goal_vic)
                        self._victims_rescued_by_human.append(self._goal_vic)
                    self._carrying_together = True
                    self._phase = Phase.FIND_NEXT_GOAL
                # When rescuing mildly injured victims alone, pick the victim up and plan the path to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._phase = Phase.PLAN_PATH_TO_DROPPOINT
                    if self._goal_vic not in self._collected_victims:
                        self._collected_victims.append(self._goal_vic)
                        self._victims_rescued_by_agent.append(self._goal_vic)
                        self._victims_rescued_by_human.append(self._goal_vic)
                    self._carrying = True
                    return CarryObject.__name__, {'object_id': self._found_victim_logs[self._goal_vic]['obj_id'],
                                                  'human_name': self._human_name}

            if Phase.PLAN_PATH_TO_DROPPOINT == self._phase:
                self._navigator.reset_full()
                # Plan the path to the drop zone
                self._navigator.add_waypoints([self._goal_loc])
                # Follow the path to the drop zone
                self._phase = Phase.FOLLOW_PATH_TO_DROPPOINT

            if Phase.FOLLOW_PATH_TO_DROPPOINT == self._phase:
                # Communicate that the agent is transporting a mildly injured victim alone to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._send_message('Transporting ' + self._goal_vic + ' to the drop zone.', 'RescueBot')
                self._state_tracker.update(state)
                # Follow the path to the drop zone
                action = self._navigator.get_move_action(self._state_tracker)
                if action is not None:
                    return action, {}
                # Drop the victim at the drop zone
                self._phase = Phase.DROP_VICTIM

            if Phase.DROP_VICTIM == self._phase:
                # Communicate that the agent delivered a mildly injured victim alone to the drop zone
                if 'mild' in self._goal_vic and self._rescue == 'alone':
                    self._send_message('Delivered ' + self._goal_vic + ' at the drop zone.', 'RescueBot')
                # Identify the next target victim to rescue
                self._phase = Phase.FIND_NEXT_GOAL
                self._rescue = None
                self._current_door = None
                self._tick = state['World']['nr_ticks']
                self._carrying = False
                # Drop the victim on the correct location on the drop zone
                return Drop.__name__, {'human_name': self._human_name}

    def _get_drop_zones(self, state):
        '''
        @return list of drop zones (their full dict), in order (the first one is the
        place that requires the first drop)
        '''
        places = state[{'is_goal_block': True}]
        places.sort(key=lambda info: info['location'][1])
        zones = []
        for place in places:
            if place['drop_zone_nr'] == 0:
                zones.append(place)
        return zones

    def _process_messages(self, state, teamMembers, condition):
        '''
        process incoming messages received from the team members
        '''
        trustBeliefs = self._loadCurrentBeliefs(self._team_members, self._folder)

        receivedMessages = {}
        # Create a dictionary with a list of received messages from each team member
        for member in teamMembers:
            receivedMessages[member] = []
        for mssg in self.received_messages:
            for member in teamMembers:
                if mssg.from_id == member:
                    receivedMessages[member].append(mssg.content)
        # Check the content of the received messages
        for mssgs in receivedMessages.values():
            for msg in mssgs:
                # If a received message involves team members searching areas, add these areas to the memory of areas that have been explored
                if msg.startswith("Search:"):
                    area = 'area ' + msg.split()[-1]
                    if area not in self._searched_rooms:
                        self._searched_rooms.append(area)
                        self._rooms_searched_by_human.append(area)
                        if self._human_name in trustBeliefs.keys():
                            trustBeliefs[self._human_name]['search']['competence'] += self.WEIGHT_HUMAN_SEARCHES_AREA
                        self._interactions += 1
                # If a received message involves team members finding victims, add these victims and their locations to memory
                if msg.startswith("Found:"):
                    if self._human_name in trustBeliefs.keys():
                        trustBeliefs[self._human_name]['search']['competence'] += self.WEIGHT_HUMAN_FINDS_VICTIM
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        foundVic = ' '.join(msg.split()[1:4])
                    else:
                        foundVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if loc not in self._searched_rooms:
                        self._interactions += 1
                        self._searched_rooms.append(loc)
                    # Add the victim and its location to memory
                    if foundVic not in self._found_victims:
                        self._found_victims.append(foundVic)
                        self._found_victim_logs[foundVic] = {'room': loc}
                    if foundVic in self._found_victims and self._found_victim_logs[foundVic]['room'] != loc:
                        self._found_victim_logs[foundVic] = {'room': loc}
                    # Decide to help the human carry a found victim when the human's condition is 'weak'
                    if condition == 'weak':
                        self._rescue = 'together'
                    # Add the found victim to the to do list when the human's condition is not 'weak'
                    if 'mild' in foundVic and condition != 'weak':
                        self._todo.append(foundVic)
                # If a received message involves team members rescuing victims, add these victims and their locations to memory
                if msg.startswith('Collect:'):
                    # Identify which victim and area it concerns
                    if len(msg.split()) == 6:
                        collectVic = ' '.join(msg.split()[1:4])
                    else:
                        collectVic = ' '.join(msg.split()[1:5])
                    loc = 'area ' + msg.split()[-1]
                    # Add the area to the memory of searched areas
                    if loc not in self._searched_rooms:
                        self._searched_rooms.append(loc)
                    # Add the victim and location to the memory of found victims
                    if collectVic not in self._found_victims:
                        self._found_victims.append(collectVic)
                        self._found_victim_logs[collectVic] = {'room': loc}
                    if collectVic in self._found_victims and self._found_victim_logs[collectVic]['room'] != loc:
                        self._found_victim_logs[collectVic] = {'room': loc}
                    # Add the victim to the memory of rescued victims when the human's condition is not weak
                    if condition != 'weak' and collectVic not in self._collected_victims:
                        self._collected_victims.append(collectVic)
                        if self._human_name in trustBeliefs.keys():
                            trustBeliefs[self._human_name]['rescue']['competence'] += self.WEIGHT_HUMAN_RESCUES_VICTIM
                        self._interactions += 1
                        self._victims_rescued_by_human.append(collectVic)
                    # Decide to help the human carry the victim together when the human's condition is weak
                    if condition == 'weak':
                        self._rescue = 'together'
                # If a received message involves team members asking for help with removing obstacles, add their location to memory and come over
                if msg.startswith('Remove:'):
                    # Come over immediately when the agent is not carrying a victim
                    if not self._carrying:
                        self._interactions += 1
                        # Identify at which location the human needs help
                        area = 'area ' + msg.split()[-1]
                        self._door = state.get_room_doors(area)[0]
                        self._doormat = state.get_room(area)[-1]['doormat']
                        if area in self._searched_rooms:
                            self._searched_rooms.remove(area)
                        # Clear received messages (bug fix)
                        self.received_messages = []
                        self.received_messages_content = []
                        self._moving = True
                        self._remove = True
                        if self._waiting and self._recent_vic:
                            self._todo.append(self._recent_vic)
                        self._waiting = False
                        # Let the human know that the agent is coming over to help
                        self._send_message(
                            'Moving to ' + str(self._door['room_name']) + ' to help you remove an obstacle.',
                            'RescueBot')
                        # Plan the path to the relevant area
                        self._phase = Phase.PLAN_PATH_TO_ROOM
                    # Come over to help after dropping a victim that is currently being carried by the agent
                    else:
                        area = 'area ' + msg.split()[-1]
                        self._send_message('Will come to ' + area + ' after dropping ' + self._goal_vic + '.',
                                          'RescueBot')
            # Store the current location of the human in memory
            if mssgs and mssgs[-1].split()[-1] in ['1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13',
                                                   '14']:
                self._human_loc = int(mssgs[-1].split()[-1])
        if self._human_name in trustBeliefs.keys():
            self._trustBelief(self._team_members, trustBeliefs, self._folder, self._received_messages)


    def _loadBelief(self, members, folder):
        '''
        Loads trust belief values if agent already collaborated with human before, otherwise trust belief values are initialized using default values.
        '''
        # Create a dictionary with trust values for all team members
        trustBeliefs = {}
        # Set a default starting trust value
        default = 0
        trustfile_header = []
        trustfile_contents = []
        # Check if agent already collaborated with this human before, if yes: load the corresponding trust values, if no: initialize using default trust values
        with open(folder + '/beliefs/allTrustBeliefs.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            header = next(reader)  # Read the header row

            for row in reader:
                # Retrieve trust values
                if row and row[0] == self._human_name:
                    name = row[0]
                    task = row[1]  # Get the task
                    if name not in trustBeliefs:
                        trustBeliefs[name] = {}  # Initialize a dictionary for the human
                    trustBeliefs[name][task] = {
                        'competence': float(row[2]),
                        'willingness': float(row[3])
                    }
        # If no prior beliefs exist for the human, initialize them
        if self._human_name not in trustBeliefs:
            trustBeliefs[self._human_name] = {
                'rescue': {'competence': default, 'willingness': default},
                'remove': {'competence': default, 'willingness': default},
                'search': {'competence': default, 'willingness': default}
            }
        with open(folder + '/beliefs/interactions.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            header = next(reader)
            for row in reader:
                if row and row[0] == self._human_name:
                    self._interactions = row[1]
                    break

        if not self._interactions:
            self._interactions = 1

        return trustBeliefs

    def _loadCurrentBeliefs(self, members, folder):
        trust_beliefs = {}
        with open(folder + '/beliefs/currentTrustBelief.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            header = next(reader)

            for row in reader:
                if row and row[0] == self._human_name:
                    name = row[0]
                    task = row[1]
                    if name not in trust_beliefs:
                        trust_beliefs[name] = {}
                    trust_beliefs[name][task] = {
                        'competence': float(row[2]),
                        'willingness': float(row[3])
                    }
        return trust_beliefs


    def _trustBelief(self, members, trustBeliefs, folder, receivedMessages):
        '''
        Baseline implementation of a trust belief. Creates a dictionary with trust belief scores for each team member, for example based on the received messages.
        '''

        # WE DO NOT NEED TO UPDATE THE TRUST ON RECEIVED MESSAGES HERE

        for task in ['rescue','search','remove']:
            trustBeliefs[self._human_name][task]['competence'] = max(-1, min(1, trustBeliefs[self._human_name][task]['competence']))
            trustBeliefs[self._human_name][task]['willingness'] = max(-1, min(1, trustBeliefs[self._human_name][task]['willingness']))
        # Save current trust belief values so we can later use and retrieve them to add to a csv file with all the logged trust belief values
        with open(folder + '/beliefs/currentTrustBelief.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['name', 'task','competence', 'willingness'])
            for task in ['rescue','search','remove']:
                csv_writer.writerow([self._human_name, task,trustBeliefs[self._human_name][task]['competence'],
                                 trustBeliefs[self._human_name][task]['willingness']])

        return trustBeliefs

    def _saveTrustBeliefs(self, members, folder):
        trustBeliefs = self._loadCurrentBeliefs(members, folder)
        allBeliefs = {}
        with open(folder + '/beliefs/allTrustBeliefs.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            header = next(reader)  # Read the header row

            for row in reader:
                if row:
                    name = row[0]
                    task = row[1]
                    if name not in allBeliefs:
                        allBeliefs[name] = {}
                    allBeliefs[name][task] = {
                        'competence': float(row[2]),
                        'willingness': float(row[3])
                    }

        allBeliefs[self._human_name] = trustBeliefs[self._human_name]

        with open(folder + '/beliefs/allTrustBeliefs.csv', mode='w') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['name', 'task', 'competence', 'willingness'])
            for name in allBeliefs.keys():
                for task in allBeliefs[name].keys():
                    csv_writer.writerow([name, task, allBeliefs[name][task]['competence'], allBeliefs[name][task]['willingness']])

        all_interactions = {}
        with open(folder + '/beliefs/interactions.csv') as csvfile:
            reader = csv.reader(csvfile, delimiter=';', quotechar="'")
            header = next(reader)
            for row in reader:
                if row:
                    if row[0] == self._human_name:
                        all_interactions[self._human_name] = self._interactions
                    else:
                        all_interactions[row[0]] = row[1]

        with open(folder + '/beliefs/interactions.csv', mode='w') as csvfile:
            csv_writer = csv.writer(csv_file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(['name', 'interactions'])
            for name in all_interactions.keys():
                csv_writer.writerow([name, all_interactions[name]])


    def _send_message(self, mssg, sender):
        '''
        send messages from agent to other team members
        '''
        msg = Message(content=mssg, from_id=sender)
        if msg.content not in self.received_messages_content and 'Our score is' not in msg.content:
            self.send_message(msg)
            self._send_messages.append(msg.content)
        # Sending the hidden score message (DO NOT REMOVE)
        if 'Our score is' in msg.content:
            self.send_message(msg)

    def _getClosestRoom(self, state, objs, currentDoor):
        '''
        calculate which area is closest to the agent's location
        '''
        agent_location = state[self.agent_id]['location']
        locs = {}
        for obj in objs:
            locs[obj] = state.get_room_doors(obj)[0]['location']
        dists = {}
        for room, loc in locs.items():
            if currentDoor != None:
                dists[room] = utils.get_distance(currentDoor, loc)
            if currentDoor == None:
                dists[room] = utils.get_distance(agent_location, loc)

        return min(dists, key=dists.get)

    def _efficientSearch(self, tiles):
        '''
        efficiently transverse areas instead of moving over every single area tile
        '''
        x = []
        y = []
        for i in tiles:
            if i[0] not in x:
                x.append(i[0])
            if i[1] not in y:
                y.append(i[1])
        locs = []
        for i in range(len(x)):
            if i % 2 == 0:
                locs.append((x[i], min(y)))
            else:
                locs.append((x[i], max(y)))
        return locs
