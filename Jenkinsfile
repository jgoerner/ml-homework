pipeline {
  agent any
  stages {
    stage('A') {
      steps {
        echo 'Hello'
      }
    }
    stage('B') {
      parallel {
        stage('B') {
          steps {
            echo 'B'
          }
        }
        stage('C') {
          agent any
          steps {
            echo 'C'
          }
        }
      }
    }
    stage('Z') {
      steps {
        echo 'Z'
      }
    }
  }
}