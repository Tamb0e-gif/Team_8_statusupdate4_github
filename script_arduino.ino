const int STEP_PIN = 2;
const int DIR_PIN  = 3;
const int EN_PIN   = 4;

void setup() {
  pinMode(STEP_PIN, OUTPUT);
  pinMode(DIR_PIN, OUTPUT);
  pinMode(EN_PIN, OUTPUT);

  digitalWrite(STEP_PIN, LOW); // begin te stappen
  digitalWrite(DIR_PIN, LOW); // richting draai
  digitalWrite(EN_PIN, LOW);  // aan/uit motor

  delay(500);


  for (int i = 0; i < 64; i++) {
    digitalWrite(STEP_PIN, HIGH);
    delayMicroseconds(5);
    digitalWrite(STEP_PIN, LOW);
    delayMicroseconds(30000);
  }
}

void loop() {
}