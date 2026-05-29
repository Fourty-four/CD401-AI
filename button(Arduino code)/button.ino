#include <ArduinoBLE.h>

const int BUTTON_PIN = A0;

// ===============================
// Button ID
// ===============================
// 0 = NONE
// 1 = YELLOW
// 2 = RED
// 3 = BLUE
// 4 = GREEN
// 5 = WHITE

const int BUTTON_NONE   = 0;
const int BUTTON_YELLOW = 1;
const int BUTTON_RED    = 2;
const int BUTTON_BLUE   = 3;
const int BUTTON_GREEN  = 4;
const int BUTTON_WHITE  = 5;

// ===============================
// Measured ADC center values
// ===============================
// 햄이 측정한 값
const int CENTER_YELLOW = 0;
const int CENTER_RED    = 565;
const int CENTER_BLUE   = 1314;
const int CENTER_GREEN  = 2029;
const int CENTER_WHITE  = 2973;

// 아무 버튼도 안 눌렀을 때 값
// 실제로 안 눌렀을 때 Serial Monitor에 뜨는 값으로 바꿔도 됨
const int CENTER_NONE   = 4095;

// ADC 값 흔들림 허용 범위
// 2028, 2029, 2030처럼 살짝 흔들리는 건 이 범위 안에서 처리됨
const int TOLERANCE = 100;

// ===============================
// BLE UUID
// ===============================
BLEService keypadService("19B10000-E8F2-537E-4F6C-D104768A1214");

BLEByteCharacteristic buttonChar(
  "19B10001-E8F2-537E-4F6C-D104768A1214",
  BLERead | BLENotify
);

BLEUnsignedShortCharacteristic rawAdcChar(
  "19B10002-E8F2-537E-4F6C-D104768A1214",
  BLERead | BLENotify
);

// ===============================
// Debounce variables
// ===============================
int stableButton = BUTTON_NONE;
int lastReadButton = BUTTON_NONE;

unsigned long lastChangeTime = 0;
const unsigned long DEBOUNCE_MS = 50;

unsigned long lastRawSendTime = 0;
const unsigned long RAW_SEND_INTERVAL_MS = 100;

// ===============================
// Button classification
// ===============================
int classifyButton(int adc) {
  int centers[6] = {
    CENTER_NONE,
    CENTER_YELLOW,
    CENTER_RED,
    CENTER_BLUE,
    CENTER_GREEN,
    CENTER_WHITE
  };

  int ids[6] = {
    BUTTON_NONE,
    BUTTON_YELLOW,
    BUTTON_RED,
    BUTTON_BLUE,
    BUTTON_GREEN,
    BUTTON_WHITE
  };

  int bestIndex = 0;
  int bestDiff = abs(adc - centers[0]);

  for (int i = 1; i < 6; i++) {
    int diff = abs(adc - centers[i]);

    if (diff < bestDiff) {
      bestDiff = diff;
      bestIndex = i;
    }
  }

  if (bestDiff > TOLERANCE) {
    return BUTTON_NONE;
  }

  return ids[bestIndex];
}

const char* buttonName(int button) {
  switch (button) {
    case BUTTON_YELLOW:
      return "YELLOW";
    case BUTTON_RED:
      return "RED";
    case BUTTON_BLUE:
      return "BLUE";
    case BUTTON_GREEN:
      return "GREEN";
    case BUTTON_WHITE:
      return "WHITE";
    case BUTTON_NONE:
    default:
      return "NONE";
  }
}

void setup() {
  Serial.begin(115200);


  analogReadResolution(12);  // Nano 33 BLE Sense Rev2: 0~4095

  pinMode(BUTTON_PIN, INPUT);

  Serial.println("Starting BLE keypad controller...");

  if (!BLE.begin()) {
    Serial.println("BLE start failed");
    while (1);
  }

  BLE.setLocalName("Nano33_Keypad");
  BLE.setDeviceName("Nano33_Keypad");

  BLE.setAdvertisedService(keypadService);

  keypadService.addCharacteristic(buttonChar);
  keypadService.addCharacteristic(rawAdcChar);

  BLE.addService(keypadService);

  buttonChar.writeValue((byte)BUTTON_NONE);
  rawAdcChar.writeValue((uint16_t)0);

  BLE.advertise();

  Serial.println("BLE keypad controller started");
  Serial.println("Device name: Nano33_Keypad");
  Serial.println("Waiting for BLE connection...");
}

void loop() {
  BLEDevice central = BLE.central();

  if (central) {
    Serial.print("Connected to: ");
    Serial.println(central.address());

    while (central.connected()) {
      int adc = analogRead(BUTTON_PIN);
      int currentButton = classifyButton(adc);

      // Debounce 처리
      if (currentButton != lastReadButton) {
        lastReadButton = currentButton;
        lastChangeTime = millis();
      }

      if ((millis() - lastChangeTime) > DEBOUNCE_MS) {
        if (currentButton != stableButton) {
          stableButton = currentButton;

          // BLE로 버튼 값 전송
          buttonChar.writeValue((byte)stableButton);

          // BLE로 raw ADC 값도 전송
          rawAdcChar.writeValue((uint16_t)adc);

          // Serial Monitor 출력
          Serial.print("ADC: ");
          Serial.print(adc);
          Serial.print(" | Button ID: ");
          Serial.print(stableButton);
          Serial.print(" | Button: ");
          Serial.println(buttonName(stableButton));
        }
      }

      // Raw ADC 값은 주기적으로 전송
      if (millis() - lastRawSendTime > RAW_SEND_INTERVAL_MS) {
        rawAdcChar.writeValue((uint16_t)adc);
        lastRawSendTime = millis();
      }

      delay(10);
    }

    Serial.print("Disconnected from: ");
    Serial.println(central.address());
    Serial.println("Waiting for BLE connection...");
  }
}
