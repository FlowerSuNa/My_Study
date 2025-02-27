# JDK, JRE, JVM

## JDK, JRE, JVM 관계

![image](https://github.com/user-attachments/assets/1ae2eb33-9d7c-4928-a6cd-0a42a02526f4)

<br>

## JDK (Java Development Kit)

- 개발자들이 자바로 개발하는 데 사용되는 SDK(Software Development Kit)

- JDK는 JRE, JVM을 모두 포함하고, 자바 개발에 필요로 한 development tools를 포함함

### Java SE (Java Standard Edition)

- 가장 기본이 되는 표준 에디션

- 가장 기본적인 클래스 패키지로 구성

- Java 언어의 핵심 기능 제공

- PC에 설치해서 사용할 수 있는 모든 개발 프로그램 사용 가능

### Java EE (Java Enterprise Edition)

- 대규모 기업용 에디션

- SE 확장판 ➡️ 대형 네트워크 환경 프로그램 개발시 사용

### Java ME (Java Micro Edition)

- SE를 라이트하게 만든 에디션

- 피처폰, 셉톱박스, 프린터와 같은 작은 임베디드 기기를 다루는데 사용

### JavaFX

- 가볍고 GUI를 제공하는 에디션

- 고성능 그래픽 하드웨어와 미디어 엔진 API를 제공해주어 프로그램 성능에 신경써야 하는 분야에서 사용

<br>

## JRE (Java Runtime Environment)

- 자바 프로그램을 실행(동작)시킬 때 필요한 라이브러리를 함께 묶어서 배포되는 패키지임

- Java로 프로그램을 직접 개발하려면 JDK가, Java 프로그램을 실행시키려면 JRE가 필요함

- 자바 런타임 환경에서 사용하는 프로퍼티 세팅이나 리소스 파일(jar 파일)을 가지고 있음

- JDK를 설치하면 함께 설치됨

<br>

## JVM (Java Virtual Machine)

- Java로 작성된 모든 프로그램은 JVM에서만 실행될 수 있으므로, Java 프로그램을 실행하기 위해서는 반드시 JVM이 필요함

- Java 언어가 인기있는 핵심 이유는 JVM 때문임

- JVM을 사용함으로써 Java 프로그램은 모든 플랫폼에서 제약없이 동작함 <br>➡️ OS에 종속적이지 않음 (이식성 높음)
    -  이러한 장점 때문에 Java가 아닌 다른 언어도 클래스 파일만 있다면 JVM을 사용할 수 있게 개발되고 있음
    - 다만, JVM을 사용하면 한 단계 더 과정을 거치게 되므로, 실행속도가 느려짐 <br>➡️ 보완을 위해 JIT 컴파일러를 사용하여 필요한 부분만 기계어로 바꾸어 줌으로써 성능 향상을 했지만, C언어의 실행속도는 따라잡지 못함 <br>➡️ 게임이나 임베디드에서 C계열 언어를 사용하는 이유임

- 단, JVM은 운영체제에 종속적이므로, 운영체제에 맞는 JVM을 설치해야 함

<br>

## Reference

- [링크](https://inpa.tistory.com/entry/JAVA-%E2%98%95-JDK-JRE-JVM-%EA%B0%9C%EB%85%90-%EA%B5%AC%EC%84%B1-%EC%9B%90%EB%A6%AC-%F0%9F%92%AF-%EC%99%84%EB%B2%BD-%EC%B4%9D%EC%A0%95%EB%A6%AC)
