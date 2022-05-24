###########################################################################################################
#    Python script for finding the shortest route between two stations in 'minutes' and plotting the
#    easiest route to travel between the stations.
#    The user has the option to indicate which lines are not in service and the corresponding stations.
#    The program shows the route accordingly.
###########################################################################################################


# Importing required libraries for the program execution

import csv  # for reading the .csv formatted files
import pandas as pd  # for reading the .csv formatted files and handling the data for relating
import matplotlib.pyplot as plt  # for plotting the graph of shortest route
import numpy as np  # for handling data series
from tkinter import *  # for implementing the GUI application


# Store the object data of rail lines in 'class Lines'.
class Lines:
    def __init__(self, line_id, node_1, node_2, weight, connect):
        self.line_id = line_id  # rail line id
        self.node_1 = node_1  # rail start-end line id
        self.node_2 = node_2  # rail start-end line id
        self.weight = weight  # time weight
        self.connect = connect
        self.plot_points = None


# Store the object data of rail stations in 'class Nodes'
class Nodes:
    visited = bool  # class global

    def __init__(self, node_name, node_id, station_tuple, node_lat, node_long):
        self.node_name = node_name  # station name
        self.node_id = node_id  # station id
        self.station_tuple = station_tuple  # tuple of ids (station and line)
        self.node_lat = node_lat  # latitude coordinate
        self.node_long = node_long  # longitude coordinate
        self.visited = False  # return visit status (type boolean)
        self.near_nodes = []  # list of nearby stations
        self.lines = []  # list of lines


# function to read, handle and generate relational data values with the given .csv formatted files
# files used :-
# 'londonstations.csv'
# 'londonlines.csv'
# 'londonconnections.csv'
def generate_data(remove_lines_list=[], remove_stations_list=[]):
    global station_ids, t_inf

    station_data = pd.read_csv('londonstations.csv').set_index('id', drop=False)  # read the station data

    station_data = station_data.loc[~(station_data['name'].isin(remove_stations_list))]  # remove stations
    station_data = station_data.rename(columns={'id': 's_id'})  # drop id column and replace with s_id
    st_ids = station_data['s_id'].tolist()

    station_ids = station_data[['s_id', 'name']].set_index('name', drop=True,
                                                           append=False)  # table of indexed station names

    lines = pd.read_csv('londonlines.csv')  # read the lines data
    remove_line_ids = lines.loc[lines['name'].isin(remove_lines_list)]['line_id'].tolist()  # remove lines

    connection_data = pd.read_csv('londonconnections.csv')  # read the connecting lines data

    # get connections after removing stations and lines:
    connection_data = connection_data.loc[
        (connection_data['station1'].isin(st_ids)) & (connection_data['station2'].isin(st_ids)) & ~(
            connection_data['line_id'].isin(remove_line_ids))]

    # merge the station and connecting lines data:
    data_combo = connection_data.merge(station_data, how='left', left_on='station1', right_on='s_id')
    data_combo['line_no'] = range(len(data_combo))

    # dictionary for storing the all station and line data
    node_data = dict()
    line_data = dict()

    # initiate the dictionary of nodes and lines data by calling the class Nodes and class Lines:
    for n in range(len(data_combo)):
        station_dict = dict(data_combo.loc[n])
        node_data[tuple([station_dict['station1'], station_dict['line_id']])] = Nodes(station_dict['name'],
                                                                                      station_dict['station1'], tuple(
                [station_dict['station1'], station_dict['line_id']]), station_dict['latitude'],
                                                                                      station_dict['longitude'])
        node_data[tuple([station_dict['station2'], station_dict['line_id']])] = Nodes(
            station_data.loc[station_dict['station2']]['name'], station_dict['station2'],
            tuple([station_dict['station2'], station_dict['line_id']]),
            station_data.loc[station_dict['station2']]['latitude'],
            station_data.loc[station_dict['station2']]['longitude'])
        line_data[n] = Lines(station_dict['line_no'], tuple([station_dict['station1'], station_dict['line_id']]),
                             tuple([station_dict['station2'], station_dict['line_id']]), station_dict['time'],
                             station_dict['line_id'])

    # merge the list of station and line ids and update the list:
    for i in range(len(data_combo)):
        station_dict = dict(data_combo.iloc[i])
        node_add_1 = node_data[tuple([station_dict['station1'], station_dict['line_id']])]
        node_add_2 = node_data[tuple([station_dict['station2'], station_dict['line_id']])]
        node_add_1.lines.append(i)
        node_add_2.lines.append(i)
        node_add_1.near_nodes.append(tuple([station_dict['station2'], station_dict['line_id']]))
        node_add_2.near_nodes.append(tuple([station_dict['station1'], station_dict['line_id']]))

    # dictionary of lines and stations with multiple connections:
    s_line = [[] for i in st_ids]
    station_tuple_list = dict(zip(st_ids, s_line))

    # update the station id with all connecting lines:
    for stat_id, node_value in node_data.items():
        station_tuple_list[stat_id[0]].append(stat_id)

    # indexing the station and line id tuple to dictionary of station ids:
    # assigning the station, line and time data to Lines object:
    t_inf = 0
    cod = 1000
    for terminal in station_tuple_list.values():
        for s_1 in terminal:
            for s_2 in terminal:
                if s_1 != s_2:
                    node_data[s_1].near_nodes.append(s_2)
                    node_data[s_1].lines.append(cod)
                    line_data[cod] = Lines(cod, s_1, s_2, t_inf, None)
                    cod = cod + 1

    # read the lines data using csv module and parsing the rail line network
    networks = list()
    s_no = list()
    line_in = open('londonlines.csv')
    read_file = csv.DictReader(line_in)

    # looping through to line data to parse the selected lines by the user
    for station_dict in read_file:
        if station_dict['line_id'] in remove_line_ids:
            continue
        networks.append(station_dict)
        s_no.append(station_dict['line_id'])
        s_no = [int(line) for line in s_no]

    lines = pd.DataFrame(dict(l=networks), index=s_no)
    return node_data, line_data, lines


# create the network of routes between stations and plot using the 'matplotlib module'
class Graph:
    def __init__(self, node_data, line_data, directed=False):
        self.directed = directed
        self.node_data = node_data  # station data
        self.line_data = line_data  # lines data

    def get_node(self, node_id):
        return self.node_data[node_id]  # returns station id

    def get_near_nodes(self, node_id):
        node = self.node_data[node_id]
        for i_no in node.near_nodes:
            return self.node_data[i_no]  # returns nearby station id

    def get_location(self, line, start_point):
        n_1 = self.get_node(line.node_1)  # Nodes object 1 (start)
        n_2 = self.get_node(line.node_2)  # Nodes object 2 (destination)

        # check in node objects for starting and destination points
        if start_point == n_1.station_tuple:
            line.plot_points = dict(start_lat=n_1.node_lat, start_long=n_1.node_long, dest_lat=n_2.node_lat,
                                    dest_long=n_2.node_long)
        if start_point == n_2.station_tuple:
            line.plot_points = dict(start_lat=n_2.node_lat, start_long=n_2.node_long, dest_lat=n_1.node_lat,
                                    dest_long=n_1.node_long)
        return line.plot_points

    # list of lists containing route line ids
    r_v = list()

    def get_route_from(self, node):
        for line_id in node.lines:
            r_v = self.line_data[line_id]
        route_in = [r_v]
        return route_in

    # return route after check
    def get_route_between(self, node_1, node_2):
        if node_1.station_tuple in node_2.near_nodes:
            for line_id in node_1.lines:
                line = self.line_data[line_id]
                if node_2.station_tuple in [line.node_2, line.node_1] and node_1.station_tuple in [line.node_2,
                                                                                                   line.node_1]:
                    return line
        else:
            return 'Sorry! There is no lines connecting the stations.'

    # returns the route using the shortest route finding algorithm called 'Dijkstra's Algorithm'
    def get_route(self, start_point, end_point):
        global next_key, station_ids, t_inf

        # get the starting node
        s_point = station_ids.loc[start_point]['s_id']
        for s in self.node_data:
            if s[0] == s_point:
                start_point = self.get_node(s)
                break

        # get the destination node
        e_point = station_ids.loc[end_point]['s_id']
        for s in self.node_data:
            if s[0] == e_point:
                end_point = self.get_node(s)
                break

        # initialise the start point to 0 and all nearby connecting nodes to infinity (any large value)
        d_v = list()
        for (i, vertex) in self.node_data.items():
            d_v.append((i, 500))
        unvisited = dict(d_v)
        unvisited[start_point.station_tuple] = 0
        visited = []
        current = start_point
        routes = {current.station_tuple: [{'name': current.node_name, 'station_id': current.station_tuple}]}
        latest = None

        # visit the nearby nodes to find the least weight of lines
        while end_point.visited == False:

            for near_id in current.near_nodes:
                near_node = self.get_node(near_id)
                near_line = self.get_route_between(current, near_node)  # []
                if near_line.weight == t_inf and (
                        near_line.node_1 == start_point.station_tuple or near_line.node_2 == end_point.station_tuple):
                    near_line.weight = 0

                if near_node.station_tuple in unvisited:

                    # assigning the line weights after check
                    if unvisited[near_node.station_tuple] > (unvisited[current.station_tuple] + near_line.weight):
                        unvisited[near_node.station_tuple] = unvisited[current.station_tuple] + near_line.weight

                        # dictionary 'routes' containing values of route station and line data
                        routes[near_node.station_tuple] = routes[current.station_tuple][:]
                        routes[near_node.station_tuple].append(
                            {"name": near_node.node_name, 'station_id': near_node.station_tuple})

            current.visited = True
            visited.append(unvisited.pop(current.station_tuple))
            for s, overall_time in unvisited.items():
                if overall_time == min(unvisited.values()):
                    next_key = s
            current = self.get_node(next_key)

        # function to plot the route
        def plot_route(route_to_end=routes[end_point.station_tuple]):
            global lines

            # set the maximum and minimum coordinate value
            coordinates_btw = dict(zip(['min_lat', 'max_lat', 'min_long', 'max_long'], [2000, -2000, 2000, -2000]))
            lc = None

            # returns the value of colour from the lines data
            def line_colours(line_id):
                self.line_id = line_id
                if line_id == None:
                    return '#999999'
                else:
                    return str('#' + lines.l[line_id]['colour'])

            # set the graph size
            plt.figure(figsize=(5, 5))
            plt.title('Route Mapper')
            plot_g = plt.subplot()

            # label the line names in graph by colour
            def points(line_id):
                self.line_id = line_id
                if line_id != None:
                    return str(lines.l[line_id]['name'])
                else:
                    return None

            # loop through a list of dictionaries, one for each station on the selected route
            for n in range(1, len(route_to_end)):
                node_1 = self.get_node(route_to_end[n - 1]['station_id'])
                node_2 = self.get_node(route_to_end[n]['station_id'])
                line = self.get_route_between(node_1, node_2)
                edge_coordinates = self.get_location(line, node_1.station_tuple)

                # check the coordinates of start and destination stations
                if True:
                    if line.plot_points['dest_lat'] > coordinates_btw['max_lat']:
                        coordinates_btw['max_lat'] = line.plot_points['dest_lat']
                    if line.plot_points['dest_lat'] < coordinates_btw['min_lat']:
                        coordinates_btw['min_lat'] = line.plot_points['dest_lat']
                    if line.plot_points['dest_long'] > coordinates_btw['max_long']:
                        coordinates_btw['max_long'] = line.plot_points['dest_long']
                    if line.plot_points['dest_long'] < coordinates_btw['min_long']:
                        coordinates_btw['min_long'] = line.plot_points['dest_long']
                    if line.plot_points['start_lat'] > coordinates_btw['max_lat']:
                        coordinates_btw['max_lat'] = line.plot_points['start_lat']
                    if line.plot_points['start_lat'] < coordinates_btw['min_lat']:
                        coordinates_btw['min_lat'] = line.plot_points['start_lat']
                    if line.plot_points['start_long'] > coordinates_btw['max_long']:
                        coordinates_btw['max_long'] = line.plot_points['start_long']
                    if line.plot_points['start_long'] < coordinates_btw['min_long']:
                        coordinates_btw['min_long'] = line.plot_points['start_long']

                line_colour = line_colours(line.connect)
                if line_colour == lc:
                    take = None
                else:
                    take = points(line.connect)
                lc = line_colour

                # formatting the graph axis, graph lines and point markers
                plt.plot([line.plot_points['start_long'], line.plot_points['dest_long']],
                         [line.plot_points['start_lat'], line.plot_points['dest_lat']], marker='X', linestyle='solid',
                         color=line_colour, label=take)
                plot_g.annotate(node_1.node_name, xy=(line.plot_points['start_long'], line.plot_points['start_lat']),
                                xytext=(line.plot_points['start_long'], line.plot_points['start_lat']))
                plot_g.annotate(node_2.node_name, xy=(line.plot_points['dest_long'], line.plot_points['dest_lat']),
                                xytext=(line.plot_points['dest_long'], line.plot_points['dest_lat']))

            # setting the axis values of latitudes and longitudes
            plot_g.set_xticks(
                [-0.60, -0.55, -0.50, -0.45, -0.40, -0.35, -0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10,
                 0.15, 0.20])
            plot_g.set_yticks(list(np.arange(51.40, 51.70, 0.01)))  # 51.50, 51.55, 51.60, 51.65, 51.70])
            plot_g.set_xlim((coordinates_btw['min_long'] - 0.02), (coordinates_btw['max_long'] + 0.02))
            plot_g.set_ylim((coordinates_btw['min_lat'] - 0.02), (coordinates_btw['max_lat'] + 0.02))
            plt.xlabel('Longitude Coordinates')
            plt.ylabel('Latitude Coordinates')
            plt.legend()

            plt.show()

        print(plot_route())
        return 'Shortest route between {} and {} is {} minutes away. Wish you safe journey!'.format(
            start_point.node_name, end_point.node_name, visited[-1])


# An application interface for getting input for the program, selecting route and station, displaying the time value:
# The interface is created using the Tkinter module:
class Application():
    def __init__(self):
        self.createWidgets()

    # creates application window using grid method
    def createWidgets(self):
        wel_label = Label(window, text='Welcome to London Underground Route Mapper!', fg='red', bg='cyan', font=('Arial', 22),
                          justify='left',
                          anchor=W)
        wel_label.grid(row=0, column=0, padx=8, pady=15)

        label1 = Label(window, text='Enter the current station: ', fg='blue', font=('Arial', 18), justify='left',
                       anchor=W)
        label1.grid(row=1, column=0)
        label2 = Label(window, text='Enter the destination station: ', fg='blue', font=('Arial', 18), justify="left")
        label2.grid(row=2, column=0)
        label3 = Label(window, text='Enter the stations with no service (comma-separated) ', fg='blue',
                       font=('Arial', 18), justify="left")
        label3.grid(row=3, column=0, columnspan=1)
        label4 = Label(window, text='Enter the lines with no service (comma-separated) ', fg='blue',
                       font=('Arial', 18), justify="left")
        label4.grid(row=4, column=0, columnspan=1)

        # variables to store the entered text
        s = StringVar()
        d = StringVar()
        ns = StringVar()
        nl = StringVar()

        textbox1 = Entry(window, textvariable=s, bg='yellow', fg='green', font=('Arial', 14))
        textbox1.grid(row=1, column=1)
        textbox2 = Entry(window, textvariable=d, bg='yellow', fg='green', font=('Arial', 14))
        textbox2.grid(row=2, column=1)
        textbox3 = Entry(window, textvariable=ns, bg='yellow', fg='green', font=('Arial', 14))
        textbox3.grid(row=3, column=1)
        textbox4 = Entry(window, textvariable=nl, bg='yellow', fg='green', font=('Arial', 14))
        textbox4.grid(row=4, column=1)

        # input stations from the interface
        def input():
            s_i = s.get()
            d_i = d.get()
            se_i = ns.get()
            se_l = nl.get()
            l = se_i.split(',')
            sl = se_l.split(',')

            global lines
            node_data, line_data, lines = generate_data(l, sl)
            network_data = Graph(node_data, line_data)
            label = Label(window, text=network_data.get_route(s_i.title(), d_i.title()), bg='green', fg='yellow',
                          font=('Arial', 20), wraplength=800)
            label.grid(row=10, column=0)

        button = Button(window, text='Go', command=input, fg='red', font=('Arial', 14))
        button.grid(row=6, column=1, sticky=E)

        button2 = Button(window, text='Quit', command=window.destroy, fg='red', font=('Arial', 14))
        button2.grid(row=7, column=1, sticky=E)


# calls the tkinter interface
window = Tk()
window.title('Route Mapper Application')
window.geometry('1000x500')

app = Application()
window.mainloop()