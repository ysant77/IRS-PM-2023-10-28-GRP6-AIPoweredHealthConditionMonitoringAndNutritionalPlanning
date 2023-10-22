<template>
  <VSonner position="top-center"></VSonner>
  <v-container id="chat-content">
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
import { hostname } from "@/api/index";
import { alertToast } from "@/util.js";

import { VSonner } from "vuetify-sonner";
import axios from "axios";

export default {
  components: {
    MsgCard,
    VSonner,
  },
  props: ["cid"],
  data: () => ({
    msgList: [],
    socket: undefined,
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
    generateSessionId() {
      return Date.now() + Math.floor(Math.random() * 1000);
    },

    checkLoginInfo() {
      let checkSocket = new WebSocket("ws://" + hostname + "/ws/check-login-info/");
      checkSocket.onmessage = (event)=>{
        let res = JSON.parse(event.data);
        if(res.redirect){
          setTimeout(() => {
            this.$router.push({ name: res.redirect });
          }, 500);
          return;
        }
        if(res.status){
          this.initConnection();
        }
      };
      checkSocket.onerror = (res)=>{
        alertToast('Connection failed!','error');
      }
    },

    initConnection() {
      let sessionId = this.generateSessionId();
      this.socket = new WebSocket(
        "ws://" + hostname + "/ws/chat/" + sessionId + "/"
      );
      this.socket.onmessage = (event) => {
        let data = JSON.parse(event.data);
        if (data.next) {
          this.createMsg(data.msg, "bot");
          this.createMsg(data.next, "bot");
          return;
        }
        this.createMsg(data, "bot");
      }
      this.socket.onerror = (event) => {
        alertToast('Connection to chatbot failed!','error')
      };
      this.socket.onclose = (event) => {
        alertToast('Connection to chatbot closed.','warning')
      };
    },

    createMsg(msg = undefined, sender) {
      if (typeof msg === "string") {
        msg = { text: msg };
      }
      msg.sender = sender;
      msg.disabled = false;
      if (this.msgList.length > 0) {
        this.msgList[this.msgList.length - 1].disabled = true;
      }
      this.msgList.push(msg);
      this.scrollDown();
    },

    scrollDown() {
      var el = document.getElementsByClassName("chat-line");
      el[el.length - 1].scrollIntoView();
    },

    sendMsg(text) {
      this.createMsg(text, "user");
      this.socket.send(
        JSON.stringify({
          message: text,
        })
      );
    },

    loadChat(cid) {
      if (cid === "new") {
        this.msgList = [
          {
            sender: "bot",
            text: "Hi! This is prompt message.",
            entries: ["ASDASd", "eqeqfqef"],
            options: ["opt 1", "opt 2", "opt 333"],
            table: {
              header: "diagnosis",
              content: [
                ["a", "b", "c", "d", "e", "f"],
                ["a", "b", "c", "d", "", "f"],
              ],
            },
            plan: [
              {
                meals: [
                  { name: "Dinner", items: "asfeaff", energy: "123" },
                  { name: "Dinner", items: "asfeaff", energy: "123" },
                ],
                total: [123,456]
              },
              {
                meals: [
                  { name: "Dinner", items: "asfeaff", energy: "123" },
                  { name: "Dinner", items: "asfeaff", energy: "123" },
                ],
                total: [123,456]
              },
            ],
            suffix: "diagnosis",
            confirm: ["Yes", "No"],
          },

          {
            sender: "bot",
            text: "Hi! This is prompt message.",
            dropdown: ["afeqfqededededededededededed", "qefewfwAEFRAGWRGGGGGGGGG", "wefewf"],
          },
        ];
      } else {
        // send http req for query in message saved
        // then push into this.msglist
      }
    },
  },

  beforeRouteUpdate(to, from) {
    if (to.params.cid === "new") {
      return;
    }
    this.loadChat(to.params.cid);
  },

  beforeRouteEnter(to, from, next) {
    next((vm) => {
      if (to.params.cid !== "new") {
        vm.loadChat(to.params.cid);
      }
      vm.checkLoginInfo();
    });
  },
};
</script>

<style scoped>
#chat-content {
  width: 100%;
  margin: 0;
  padding-top: 2%;
  padding-left: 3%;
  padding-right: 3%;
  padding-bottom: 15%;
}
</style>
