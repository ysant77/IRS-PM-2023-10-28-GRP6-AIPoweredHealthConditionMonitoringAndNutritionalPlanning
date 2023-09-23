<template>
  <v-container>
    <template v-for="(msg, i) in msgList" :key="i">
      <div
        class="chat-line pa-2 d-flex align-center"
        :class="msg.sender == 'bot' ? 'justify-start' : 'justify-end'"
        :style="msgStyles[msg.sender]"
      >
        <v-card
          w-auto
          :text="msg.text"
          :variant="msg.sender == 'user' ? 'tonal' : 'default'"
          elevation="2"
        ></v-card>
      </div>
    </template>

    <v-btn @click="bindBtn">Btn</v-btn>
  </v-container>
</template>

<script setup>
</script>

<script>
export default {
  data: () => ({
    msgList: [
      {
        sender: "bot",
        type: 0,
        text: "Sample message 1.",
      },
      {
        sender: "user",
        type: 0,
        text: "Sample message 2.",
      },
      {
        sender: "bot",
        type: 0,
        text: "Et accusamus provident in eius velit aut quod saepe qui quia consequatur. ",
      },
      {
        sender: "user",
        type: 0,
        text: "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum.",
      },
    ],
    msgStyles: {
      bot: {
        "margin-right": "3%",
      },
      user: {
        "margin-left": "3%",
      },
    },
  }),

  methods: {
    createMsg(text) {
      this.msgList.push({
        sender: "user",
        text: text,
      });
      this.scrollDown();
    },

    bindBtn() {
      this.createMsg('Another msg.')
    },

    scrollDown() {
      var el = document.getElementsByClassName('chat-line');
      el[el.length-1].scrollIntoView();
    }
  },

  // hook
  mounted() {
    this.createMsg('Msg created on load.');
  },
};
</script>

<style scoped>
.v-container {
  width: 100%;
  margin: 0;
  padding-left: 3%;
  padding-right: 3%;
  padding-bottom: 10%;
}
</style>