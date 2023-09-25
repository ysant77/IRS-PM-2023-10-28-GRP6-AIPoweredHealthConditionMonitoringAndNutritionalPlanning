<template>
  <v-layout class="rounded rounded-md">
    <v-app-bar elevation="4">
      <template v-slot:prepend>
        <v-app-bar-nav-icon @click.stop="drawer = !drawer"></v-app-bar-nav-icon>
      </template>

      <v-app-bar-title>Health ChatBot</v-app-bar-title>

      <template v-slot:append>
        <v-btn
          prepend-icon="mdi-content-save"
          variant="flat"
          @click.stop="saveChat"
        >
          Save Chat
          <!-- <v-tooltip activator="parent"> Save chat </v-tooltip> -->
        </v-btn>
        <v-btn
          prepend-icon="mdi-plus-circle"
          variant="elevated"
          @click.stop="newChat"
        >
          New Chat
        </v-btn>
      </template>
    </v-app-bar>

    <v-navigation-drawer v-model="drawer" elevation="3">
      <v-list>
        <v-list-item
          link
          v-for="item in historyList"
          :key="item.title"
          :title="item.title"
          :to="{ name: 'ChatHistory', params: { cid: item.tstamp } }"
        >
          <v-tooltip activator="parent"> TIME </v-tooltip>
        </v-list-item>
      </v-list>
    </v-navigation-drawer>

    <v-main
      class="d-flex align-center justify-center"
      style="min-height: 300px"
    >
      <router-view v-slot="{ Component }">
        <component ref="view" :is="Component"></component>
      </router-view>
    </v-main>
  </v-layout>
</template>

<script>
export default {
  data: () => ({
    drawer: null,

    historyList: null,
  }),

  methods: {
    // Call Methods of Chat view.
    saveChat() {
      this.$refs.view.saveChat();
    },

    newChat() {
      this.$route.push({name:'NewChat'});
    },

    loadHistories() {
      this.historyList = [
        {
          title: "Recent chat",
          tstamp: 1695562257580,
        },
        {
          title: "Old chat 2",
          tstamp: 1695532257580,
        },
        {
          title: "Older chat 3",
          tstamp: 1694532257580,
        },
      ];
    },
  },

  mounted() {
    this.loadHistories();
  },
};
</script>

<style scoped>
</style>