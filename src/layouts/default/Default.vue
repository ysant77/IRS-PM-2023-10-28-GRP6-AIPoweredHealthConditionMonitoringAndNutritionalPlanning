<template>
  <v-layout class="rounded rounded-md">
    <v-app-bar elevation="4">
      <template v-slot:prepend>
        <v-app-bar-nav-icon @click.stop="drawer = !drawer"></v-app-bar-nav-icon>
      </template>

      <v-app-bar-title>Chat</v-app-bar-title>

      <template v-slot:append>
        <v-btn icon="mdi-content-save">
          <v-icon></v-icon>
          <v-tooltip activator="parent"> Save chat </v-tooltip>
        </v-btn>
      </template>
    </v-app-bar>

    <v-navigation-drawer v-model="drawer" elevation="3">
      <v-list>
        <v-list :items="items"></v-list>
      </v-list>
    </v-navigation-drawer>

    <!-- <v-app-bar height="100%" location="bottom" elevation="8">
      <v-container>
        <v-textarea
          name="input-7-1"
          variant="outlined"
          rows="1"
          auto-grow
          placeholder="Input..."
        ></v-textarea>
      </v-container>
    </v-app-bar> -->

    <v-main
      class="d-flex align-center justify-center"
      style="min-height: 300px"
    >
      <router-view></router-view>
    </v-main>
  </v-layout>
</template>


<!-- <script setup>
  var drawer = null;
  var items = [
      {
        title: "Foo",
        value: "foo",
      },
      {
        title: "Bar",
        value: "bar",
      },
    ];
</script> -->

<script>
export default {
  data: () => ({
    drawer: null,
    items: [
      {
        title: "Foo",
        value: "foo",
      },
      {
        title: "Bar",
        value: "bar",
      },
    ],
  }),

  created() {
    console.log('114514!');
  },
};
</script>

<style scoped>
</style>