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
          :disabled="disabled"
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
import { hostname, axios } from "@/api.js";
import { alertToast, testdata } from "@/util.js";

import { VSonner } from "vuetify-sonner";

export default {
  components: {
    MsgCard,
    VSonner,
  },
  props: ["cid"],
  emits: ["disableSend","enableSend","updateHist"],

  data: () => ({
    disabled: false,
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
      this.session_id = Date.now() + Math.floor(Math.random() * 1000);
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
      this.socket = new WebSocket(
        "ws://" + hostname + "/ws/chat/" + this.session_id + "/"
      );
      this.socket.onmessage = (event) => {
        this.$emit("enableSend");
        this.disabled = false;
        let data = JSON.parse(event.data);
        if (data.next) {
          this.createMsg(data.msg, "bot");
          this.createMsg(data.next, "bot");
          return;
        }
        this.createMsg(data, "bot");
      };
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
      this.msgList.push(msg);
      setTimeout('this.scrollDown()',300);
    },

    scrollDown() {
      var el = document.getElementsByClassName("chat-line");
      el[el.length - 1].scrollIntoView();
    },

    sendMsg(text) {
      this.createMsg(text, "user");
      this.$emit('disableSend');
      this.disabled = true;
      this.socket.send(
        JSON.stringify({
          message: text,
        })
      );
    },

    loadChat(cid) {
      this.msgList = []
      if (cid === "new") {
        // this.msgList = testdata;
        return;
      } else {
        // send http req for query in message saved
        // then push into this.msglist
        axios.post("http://"+hostname+"/api/get-chat-msgs", {'cid':cid})
        .then((res)=>{
          if(res.data.status){
            // this.msgList = res.data.data;
            for (let i=0; i<res.data.data.length; i++){
              let obj = res.data.data[i];
              if (obj.is_user){
                this.createMsg(obj.content, 'user');
                continue;
              }else{
                obj = JSON.parse(obj.content);
              } 
              if(obj.next){
                this.createMsg(obj.msg, "bot");
                this.createMsg(obj.next, "bot");
                continue;
              }
              this.createMsg(obj, 'bot');
            };
          }else{
            alertToast('Fetch chat messages failed!','error');
          }
        })
      }
    },
  },

  beforeRouteUpdate(to, from) {
    this.loadChat(to.params.cid);
    if (to.params.cid === "new") {
      this.generateSessionId()
    }else{
        this.session_id = to.params.cid;
    }
    this.initConnection();
    this.$emit("updateHist");
  },

  beforeRouteEnter(to, from, next) {
    next((vm) => {
      // if (to.params.cid !== "new") {
      //   vm.loadChat(to.params.cid);
      // }
      vm.loadChat(to.params.cid);
      if (to.params.cid === 'new'){
        vm.generateSessionId();
      }else{
        vm.session_id = to.params.cid;
      }
      vm.checkLoginInfo();
      vm.$emit("updateHist");
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
