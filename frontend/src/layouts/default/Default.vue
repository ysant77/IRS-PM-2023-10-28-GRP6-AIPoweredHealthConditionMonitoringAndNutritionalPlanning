<template>
  <v-layout class="rounded rounded-md">
    <v-app-bar color="secondary" elevation="4">
      <template v-slot:prepend>
        <v-app-bar-nav-icon @click.stop="drawer = !drawer"></v-app-bar-nav-icon>
      </template>

      <v-app-bar-title>Health ChatBot</v-app-bar-title>

      <template v-slot:append>
        <v-btn
          prepend-icon="mdi-account-edit"
          variant="outlined"
          :to="{ name: 'info' }"
        >
          Edit my info
        </v-btn>
      </template>
    </v-app-bar>

    <v-navigation-drawer width="350" v-model="drawer" elevation="3" color="bg">
      <v-list lines="two">
        <v-list-item
          link
          size="large"
          v-for="item in historyList"
          :key="item.title"
          :title="'Chat title'"
          :subtitle="'Saved time'"
          :to="{ name: 'Chat', params: { cid: item.tstamp } }"
        >
          <!-- <v-tooltip activator="parent"> TIME </v-tooltip> -->
        </v-list-item>

        <v-btn
          block
          prepend-icon="mdi-plus"
          variant="text"
          size="large"
          class="my-2 px-3"
          color="primary"
          @click.stop="newChat"
          >New chat</v-btn
        >
      </v-list>
    </v-navigation-drawer>

    <v-app-bar location="bottom" height="auto" elevation="0" grow>
      <v-text-field
        class="mb-1 ml-3 mt-1"
        v-model="input"
        variant="outlined"
        rows="1"
        placeholder="Input here..."
        color="primary"
        @keyup.enter="sendInput"
        hide-details
        clearable
      ></v-text-field>
      <v-btn
        class="mr-10"
        icon="mdi-send"
        size="large"
        color="primary"
        @click.stop="sendInput"
      ></v-btn>
    </v-app-bar>

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
    input: "",
    historyList: null,
  }),

  methods: {
    // Call Methods of Chat view.
    saveChat() {
      this.$refs.view.saveChat();
    },

    newChat() {
      // this.$router.push({ name: "Chat", params: { cid: "new" } });
      location.reload();
    },

    loadHistories() {
      this.historyList = [
        {
          title: "Recent chat",
          tstamp: 1695562257580,
        },
        {
          title: "",
          tstamp: 1695532257580,
        },
        {
          title: "Older chat 3",
          tstamp: 1694532257580,
        },
      ];
    },

    sendInput() {
      if (this.input === "\n" || this.input === "") {
        this.input = "";
        return;
      }
      this.$refs.view.sendMsg(this.input);
      this.input = "";
    },
  },

  mounted() {
    this.loadHistories();
  },
};
</script>
