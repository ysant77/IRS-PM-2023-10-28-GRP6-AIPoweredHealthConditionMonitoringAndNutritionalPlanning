<template>
  <title>{{ chatTitle }}</title>
  <v-container>
    <div v-for="(msg, i) in msgList" :key="i">
      <div
        class="chat-line pa-2 d-flex align-center"
        :class="msg.sender == 'bot' ? 'justify-start' : 'justify-end'"
        :style="msgStyles[msg.sender]"
      >
        <MsgCard
          :msg="msg"
          @sendMsg="
            (msg) => {
              sendMsg(msg);
            }
          "
        />
      </div>
    </div>
  </v-container>
</template>

<script>
import MsgCard from "@/components/MsgCard.vue";

export default {
  components: {
    MsgCard,
  },
  props: ["cid"],
  data: () => ({
    chatTitle: "chat title",
    msgList: null,

    msgStyles: {
      bot: {
        "margin-right": "5%",
      },
      user: {
        "margin-left": "5%",
      },
    },
  }),

  methods: {
    // add new msg to list
    createMsg(text, sender) {
      this.msgList.push({
        sender: sender,
        text: text,
      });
      this.scrollDown();
    },

    // scroll to bottom
    scrollDown() {
      var el = document.getElementsByClassName("chat-line");
      el[el.length - 1].scrollIntoView();
    },

    /* Need to stop user from sending until it's done. */
    sendMsg(text) {
      this.createMsg(text, "user");
      var ret_text = this.getReply(text);
      this.createMsg(ret_text, "bot");
    },

    /* Edit this for API. */
    getReply(text) {
      return 'Reply to message "' + text + '"';
    },

    saveChat() {
      const cid = Date.now();
      const content = {
        chatTitle: this.chatTitle,
        msgList: this.msgList,
      };
      console.log("Chat saved" + this.chatTitle);
    },

    loadTitle(cid) {
      this.chatTitle = (cid==='new')? 'New Chat' : 'Chat'+cid;
    },

    loadChat(cid) {
      if (cid === "new") {
        this.msgList = [
          {
            sender: "bot",
            text: "Hi! This is prompt message.",
            options: [
              { text: "View diagnosis.", value: "View diagnosis of..." },
              { text: "Nutrition planning.", value: "Nutrition plans." },
            ],
          },
        ];
      } else {
        this.msgList = [
          {
            sender: "bot",
            text: "Hi! This is prompt message for a saved chat. " + cid,
            options: [
              { text: "View diagnosis.", value: "View diagnosis of..." },
              { text: "Nutrition planning.", value: "Nutrition plans." },
            ],
          },
          {
            sender: "user",
            text: "Sample.",
          },
        ];
      }
    },
  },

  beforeRouteUpdate(to, from) {
    this.loadTitle(to.params.cid);
    this.loadChat(to.params.cid);
  },

  beforeRouteEnter(to, from, next) {
    next((vm) => {
      vm.loadTitle(to.params.cid);
      vm.loadChat(to.params.cid);
    });
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
