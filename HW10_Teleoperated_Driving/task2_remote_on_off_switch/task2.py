"""
Goal of Task 2:
    Implement a program which simulates a remote ON-OFF switch.

Hints:
    - Compare to the code introduced in the practice session.
    - Use the web interface http://tools.emqx.io
"""


import paho.mqtt.client as mqtt


""" ! do not change anything from here ! """

sub_topic_preset = "testtopic/sub_topic"
pub_topic_preset = "testtopic/pub_topic"
broker_preset = "broker.emqx.io"
port_preset = 1883
global_pub_topic = ""
global_sub_topic = ""


def adapt_global_topic_names(sub_topic, pub_topic):
    global global_sub_topic
    global_sub_topic = sub_topic
    global global_pub_topic
    global_pub_topic = pub_topic


""" ! to here ! """


def on_connect(client, userdata, flags, rc):
    """
    This function should be the callback function, which is called, when the client connects successfully to the broker.
    """

    print("A connection has been established")
    global global_sub_topic

    # Subtask 1:
    # ToDo: Make the client subscribe to the topic 'global_sub_topic'.
    ########################
    #  Start of your code  #
    ########################

    print("Connected with result code " + str(rc))
    client.subscribe(global_sub_topic)

    ########################
    #   End of your code   #
    ########################


def on_message(client, userdata, msg):
    """
    This function should be the callback function for incoming messages on the previously defined subscription topic.
    """

    global global_pub_topic
    current_payload = msg.payload.decode("utf-8")
    print("Received a message: " + str(current_payload))

    # Subtask 2:
    # ToDo: Depending on the incoming message, different content should be published on the topic 'global_pub_topic'.
    #  If the message on the subscription_topic is 'on', the publication topic should tell 'The light is shining' once.
    #  If the message on the subscription_topic is 'off', the publication topic should tell
    #  'The light is not shining anymore' once.
    ########################
    #  Start of your code  #
    ########################

    finish_statement = 'on'
    if current_payload == finish_statement:
        client.publish(global_pub_topic, "The light is shining")
    else:
        client.publish(global_pub_topic, "The light is not shining anymore")

    ########################
    #   End of your code   #
    ########################


def main(sub_topic=sub_topic_preset, pub_topic=pub_topic_preset,
         broker=broker_preset, port=port_preset):

    adapt_global_topic_names(sub_topic, pub_topic)
    client = mqtt.Client()

    # Subtask 3:
    # ToDo: Set the callback functions which are automatically called by the client once a connection is established
    #   and once a message is received.
    ########################
    #  Start of your code  #
    ########################

    client.on_connect = on_connect
    client.on_message = on_message

    ########################
    #   End of your code   #
    ########################

    client.connect(broker, port=port, keepalive=10)
    client.loop_forever()


if __name__ == "__main__":
    main()
