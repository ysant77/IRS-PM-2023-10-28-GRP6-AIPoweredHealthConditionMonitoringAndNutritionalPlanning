<template>
  <v-layout class="rounded rounded-md">
    <default-bar></default-bar>

    <default-drawer></default-drawer>

    <v-main class="d-flex align-center justify-center" style="min-height: 300px;">
      Main Area
    </v-main>
  </v-layout>
</template>

<script setup>
  import DefaultBar from './AppBar.vue'
  import DefaultDrawer from './Drawer.vue'
</script>
