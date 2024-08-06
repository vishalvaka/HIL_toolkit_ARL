const int pressurePin1 = A0;  // Pin where the first pressure sensor is connected

const int pressurePin2 = A1;  // Pin where the second pressure sensor is connected

unsigned long previousMillis = 0;

const long interval = 100;  // Interval at which to read sensors (1 second)

int pressureCount = 0;




void setup() {

  // put your setup code here, to run once:

Serial.begin(115200);

}




void loop() {

 unsigned long currentMillis = millis();




  if (currentMillis - previousMillis >= interval) {

    previousMillis = currentMillis;

    int pressure1 = analogRead(pressurePin1);

    int pressure2 = analogRead(pressurePin2);

    pressureCount++;

    unsigned long time = currentMillis;  // Time in milliseconds




    // Send data in a simpler format to reduce overhead

   Serial.print("(");

    Serial.print(time);

    Serial.print(",");

    Serial.print(pressureCount);

    Serial.print(",");

    Serial.print(pressure1);

    Serial.print(",");

    Serial.print(pressure2);

    Serial.println(")");

  }




}

