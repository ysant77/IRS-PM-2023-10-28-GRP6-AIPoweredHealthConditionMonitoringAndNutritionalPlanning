<template>
  <v-layout ref="app" class="rounded rounded-md">
    <v-main
      class="d-flex align-center justify-center"
      style="min-height: 500px"
    >
      <v-card class="my-50 px-5 py-3" elevation="0">

          <v-btn
            prepend-icon="mdi-google"
            size="large"
            class="bg-primary text-none mx-auto"
            href="http://localhost:8000/accounts/google/login/?process=login"
          >
            Login with Google
          </v-btn>

      </v-card>
    </v-main>
  </v-layout>
</template>
