# Professor David Crandall
# B551 Elements of Artificial Intelligence
# Yue Chen, Siddharth Jayant Pathak, and Raghottam Dilip Talwai
# October 1st, 2017

# Problem 2

class Survey:

    count = 0

    def __init__(self, name, number, prefer, notPrefer):
        self.name = name
        self.number = number
        self.prefer = prefer.split(",") if prefer != "_" else []
        self.notPrefer = notPrefer.split(",") if notPrefer != "_" else []
        Survey.count += 1

    def __repr__(self):
        return self.name

def successors(state):

    successorState = []

    for eachGroup in state.assigned:
        if len(eachGroup) < 3:
            newState = State(assigned=state.assigned, people=state.people)
            newState.add_person(newState.unassigned[0], state.assigned.index(eachGroup))
            successorState.append(newState)
    newState = State(assigned=state.assigned, people=state.people)
    newState.add_person(newState.unassigned[0], len(newState.assigned))
    successorState.append(newState)

    return successorState

def is_goal(state):
    if len(state.unassigned) == 0:
        return True
    else:
        return False

class State:

    gradeTime = 0
    emailTime = 0
    meetTime = 0

    priority = 0

    people = []
    assigned = []
    unassigned = []

    def __init__(self, assigned, people):

        # Yingnan Ju helped me with 2-dimensional list copy-and-paste.
        self.assigned = []
        for eachGroup in assigned:
            self.assigned.append(list(eachGroup))
        self.update_priority()
        self.people = list(people)
        self.update_unassigned()

    def set_time(self, gradeTime, emailTime, meetTime):
        State.gradeTime = gradeTime
        State.emailTime = emailTime
        State.meetTime = meetTime

    def update_priority(self):

        assignedCount = 0
        sinkingCost = len(self.assigned) * State.gradeTime
        for group in self.assigned:
            for person in group:
                assignedCount += 1
                if person.number != len(group) and person.number != 0:
                    sinkingCost += 1
                for preferred in person.prefer:
                    if preferred not in list(group[i].name for i in range(0, len(group))):
                        sinkingCost += State.emailTime
                for notPreferred in person.notPrefer:
                    if notPreferred in list(group[i].name for i in range(0, len(group))):
                        sinkingCost += State.meetTime

        # The lowest possible remaining cost given all assigned groups have 3 people and all unassigned groups will have 3 people
        # heuristic = ((Survey.count - len(self.assigned) *3 ) // 3 + (1 if (Survey.count - len(self.assigned) * 3) % 3 > 0 else 0)) * State.gradeTime
        heuristic = math.ceil((Survey.count - len(self.assigned) * 3) / 3) * State.gradeTime

        self.priority = sinkingCost + heuristic


    def add_person(self, person, groupIndex):
        if groupIndex < len(self.assigned):
            self.assigned[groupIndex].append(person) if len(self.assigned[groupIndex]) < 3 else None
        else:
            self.assigned.append([person])
        self.update_priority()
        self.update_unassigned()

    def update_unassigned(self):
        self.unassigned.clear()
        flatAssigned = []
        for group in self.assigned:
            flatAssigned.extend(group)
        for person in self.people:
            if person not in flatAssigned:
                self.unassigned.append(person)

def solve(initialState):
    if is_goal(initialState):
        return initialState
    else:
        fringe = [initialState]
        while len(fringe) > 0:
            # print(len(fringe))
            print([state.priority for state in fringe])
            # if fringe.pop(0), djcran and kapadia will be stuck together forever :)
            for s in successors(fringe.pop()):
                if is_goal(s):
                    return s
                isInFringe = False
                for eachState in fringe:
                    if eachState.priority <= s.priority:
                        fringe.insert(fringe.index(eachState), s)
                        isInFringe = True
                        break
                if not isInFringe:
                    fringe.append(s)
        return False

# main()

import sys, copy, math

inputFileName = sys.argv[1]
gradeTime = int(sys.argv[2])
emailTime = int(sys.argv[3])
meetTime = int(sys.argv[4])

inputFile = open(inputFileName, 'r')

people = []

for line in inputFile:
    person = line.split()
    people.append(Survey(name=person[0],number=person[1], prefer=person[2], notPrefer=person[3]))

initialState = State(assigned=[], people=people)

initialState.set_time(gradeTime=gradeTime, emailTime=emailTime, meetTime=meetTime)

solution = solve(initialState)

for group in solution.assigned:
    print(group)
print(solution.priority)