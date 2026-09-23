#include <ESP8266WiFi.h>
#include <WiFiClientSecureBearSSL.h>
#include <PubSubClient.h>
#include "secrets.h"

const char* MQTT_BROKER  = "29455b01c27447b488b1ec93488ce95d.s1.eu.hivemq.cloud";
const int   MQTT_PORT    = 8883;
const char* MQTT_CLIENT  = "gesture_car_esp8266";
const char* TOPIC_CMD    = "gesturecar/command";
const char* TOPIC_STATUS = "gesturecar/status";

#define IN1  D1
#define IN2  D2
#define IN3  D5
#define IN4  D6
#define ENA  D7
#define ENB  D8
#define SPEED 700

unsigned long lastCmd  = 0;
unsigned long lastPing = 0;
const unsigned long WATCHDOG_MS = 2000;
const unsigned long PING_MS     = 5000;

BearSSL::WiFiClientSecure net;
PubSubClient mqtt(net);

void stopMotors() {
  digitalWrite(IN1, LOW); digitalWrite(IN2, LOW);
  digitalWrite(IN3, LOW); digitalWrite(IN4, LOW);
  analogWrite(ENA, 0);    analogWrite(ENB, 0);
}

// drive(l1, l2, r1, r2, speed)
// l1=1 l2=0 → left side forward
// l1=0 l2=1 → left side reverse
// r1=1 r2=0 → right side forward
// r1=0 r2=1 → right side reverse
void drive(bool l1, bool l2, bool r1, bool r2, int spd) {
  digitalWrite(IN1, l1); digitalWrite(IN2, l2);
  digitalWrite(IN3, r1); digitalWrite(IN4, r2);
  analogWrite(ENA, spd); analogWrite(ENB, spd);
}

void execute(String cmd) {
  cmd.trim(); cmd.toUpperCase();

  if (cmd == "PING") {
    Serial.println("[CMD] PING — ignored");
    return;
  }

  Serial.printf("[CMD] %s\n", cmd.c_str());

  // Corrected mapping based on observed physical behaviour:
  // Original drive calls were producing wrong directions.
  // Mapping corrected to match actual motor response.
  if      (cmd == "FORWARD") drive(1, 0, 0, 1, SPEED);
  else if (cmd == "REVERSE") drive(0, 1, 1, 0, SPEED);
  else if (cmd == "LEFT")    drive(1, 0, 1, 0, SPEED);
  else if (cmd == "RIGHT")   drive(0, 1, 0, 1, SPEED);
  else if (cmd == "STOP")    stopMotors();
  else {
    Serial.printf("[CMD] Unknown: %s\n", cmd.c_str());
    stopMotors();
  }

  lastCmd = millis();
}

void onMessage(char* topic, byte* payload, unsigned int len) {
  String msg;
  for (unsigned int i = 0; i < len; i++) msg += (char)payload[i];
  execute(msg);
}

void connectWiFi() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASS);
  Serial.print("[WIFI] Connecting");
  int t = 0;
  while (WiFi.status() != WL_CONNECTED && t < 40) {
    delay(500); Serial.print("."); t++;
  }
  if (WiFi.status() == WL_CONNECTED)
    Serial.printf("\n[WIFI] Connected — %s\n", WiFi.localIP().toString().c_str());
  else
    Serial.println("\n[WIFI] Failed");
}

void connectMQTT() {
  net.setInsecure();
  mqtt.setServer(MQTT_BROKER, MQTT_PORT);
  mqtt.setCallback(onMessage);
  mqtt.setKeepAlive(30);
  mqtt.setBufferSize(512);
  Serial.print("[MQTT] Connecting");
  int t = 0;
  while (!mqtt.connected() && t < 5) {
    if (mqtt.connect(MQTT_CLIENT, MQTT_USER, MQTT_PASS)) {
      Serial.println("\n[MQTT] ✓ Connected");
      mqtt.subscribe(TOPIC_CMD);
      mqtt.publish(TOPIC_STATUS, "ONLINE");
    } else {
      Serial.printf(" rc=%d", mqtt.state());
      delay(3000); t++;
    }
  }
}

void setup() {
  Serial.begin(115200); delay(100);
  pinMode(IN1,OUTPUT); digitalWrite(IN1,LOW);
  pinMode(IN2,OUTPUT); digitalWrite(IN2,LOW);
  pinMode(IN3,OUTPUT); digitalWrite(IN3,LOW);
  pinMode(IN4,OUTPUT); digitalWrite(IN4,LOW);
  pinMode(ENA,OUTPUT); analogWrite(ENA,0);
  pinMode(ENB,OUTPUT); analogWrite(ENB,0);

  Serial.println("\n══════════════════════════");
  Serial.println("  Gesture RC Car");
  Serial.printf ("  WiFi  : %s\n", WIFI_SSID);
  Serial.printf ("  Broker: %s\n", MQTT_BROKER);
  Serial.println("══════════════════════════");

  connectWiFi();
  if (WiFi.status() == WL_CONNECTED) connectMQTT();
  lastCmd = millis(); lastPing = millis();
}

void loop() {
  if (WiFi.status() != WL_CONNECTED) { connectWiFi(); }
  if (WiFi.status() == WL_CONNECTED && !mqtt.connected()) { connectMQTT(); }
  if (mqtt.connected()) mqtt.loop();
  if (millis() - lastCmd > WATCHDOG_MS) { stopMotors(); lastCmd = millis(); }
  if (mqtt.connected() && millis() - lastPing > PING_MS) {
    mqtt.publish(TOPIC_STATUS, "ALIVE");
    lastPing = millis();
  }
  yield();
}