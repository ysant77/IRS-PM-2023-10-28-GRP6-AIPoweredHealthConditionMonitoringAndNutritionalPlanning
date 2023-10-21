<template>
  <VSonner position="top-center"></VSonner>
  <v-layout class="rounded rounded-md">
    <v-app-bar color="secondary" elevation="4">
      <template v-slot:prepend>
        <v-app-bar-nav-icon @click.stop="drawer = !drawer"></v-app-bar-nav-icon>
      </template>

      <v-app-bar-title>Health ChatBot</v-app-bar-title>

      <template v-slot:append>
        <v-btn
          icon="mdi-bell"
          size="small"
          class="mx-5"
          variant="outlined"
          :to="{ name: 'configTelegram' }"
        >
        </v-btn>
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
          v-for="item,i in historyList"
          :title="'My chat '+ i"
          :subtitle="'Created at (UTC)' + item.time"
          :to="{ name: 'Chat', params: { cid: item.cid } }"
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
        :disabled="disabled"
      ></v-text-field>
      <v-btn
        class="mr-10"
        icon="mdi-send"
        size="large"
        color="primary"
        @click.stop="sendInput"
        :disabled="disabled"
      ></v-btn>
    </v-app-bar>

    <v-main
      class="d-flex align-center justify-center"
      style="min-height: 300px"
    >
      <router-view v-slot="{ Component }">
        <component
          ref="view"
          :is="Component"
          @disableSend="disableSend"
          @enableSend="enableSend"
          @updateHist="loadHistories"
        ></component>
      </router-view>
    </v-main>
  </v-layout>
</template>

<script>
import { hostname, axios } from "@/api.js";
import { alertToast } from "@/util.js";

import { VSonner } from "vuetify-sonner";

export default {
  components: {
    VSonner,
  },

  data: () => ({
    drawer: null,
    input: "",
    historyList: null,
    disabled: false,
  }),

  methods: {
    // Call Methods of Chat view.
    newChat() {
      this.$router.replace({ name: "Chat", params: { cid: "new" } });
    },

    loadHistories() {
      axios
        .get("http://" + hostname + "/api/curr-user-hist")
        .then((res) => {
          if (res.data.status) {
            this.historyList = res.data.data;
          } else {
            alertToast("Load chat history failed!", "error");
          }
        })
        .catch((res) => {
          alertToast("Connection failed!", "error");
        });
    },

    sendInput() {
      if (this.input === "\n" || this.input === "") {
        this.input = "";
        return;
      }
      this.$refs.view.sendMsg(this.input);
      this.input = "";
    },

    disableSend() {
      this.disabled = true;
    },

    enableSend() {
      this.disabled = false;
    },
  },

  // mounted() {
  //   this.loadHistories();
  // },
};
</script>
