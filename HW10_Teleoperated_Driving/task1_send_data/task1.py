"""
Goal of Task 1:
    Use an mqtt-client to send data to a broker periodically.

Hint:
    Use http://tools.emqx.io/ to see what's going on on your topics.
"""


import paho.mqtt.client as mqtt
import time


""" ! do not change anything from here ! """
mqtt_topic_preset = "testtopic/info"
payload_string_preset = "word"
broker_preset = "broker.emqx.io"
port_preset = 1883
""" ! to here ! """


def main(topic=mqtt_topic_preset, payload=payload_string_preset,
         broker=broker_preset, port=port_preset):

    client = mqtt.Client()
    client.connect(broker, port=port, keepalive=10)
    client.loop_start()
    while True:

        # Task:
        # ToDo: Implement the missing code which enables the function to publish the provided payload
        #  to the provided topic periodically.
        # Hints:
        #   - The interval of publish is 1 Hz and is already implemented with the time.sleep(1) function.
        #   - The provided payload has to be a string.
        ########################
        #  Start of your code  #
        ########################

        client.publish(topic, payload)

        ########################
        #   End of your code   #
        ########################

        print("Published " + str(payload) + " to the broker " + str(broker) + " at the topic " + str(topic))
        time.sleep(1)


if __name__ == "__main__":
    main()
