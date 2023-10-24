<template>
  <VSonner position="top-center"></VSonner>
  <v-container id="chat-content">
    <div v-for="msg in msgList" ref="scrollRefs">
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

    checkLoginInfo(to=null, from=null) {
      this.checkSocket = new WebSocket("ws://" + hostname + "/ws/check-login-info/");
      this.checkSocket.onmessage = (event)=>{
        let res = JSON.parse(event.data);
        if(res.redirect){
          setTimeout(() => {
            this.$router.push({ name: res.redirect });
          }, 500);
          return;
        }
        if(res.status){
          this.switchRoute(to,from);
        }
      };
      this.checkSocket.onerror = (res)=>{
        alertToast('Connection failed!','error');
      };
    },

    initConnection() {
      if (this.socket){
        this.socket.close()
      }

      this.socket = new WebSocket(
        "ws://" + hostname + "/ws/chat/" + this.session_id + "/"
      );

      this.socket.onopen = (event) => {
        this.$emit("updateHist");
      }

      this.socket.onmessage = (event) => {
        this.$emit("enableSend"); // enable user send msg
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
    },

    createMsg(msg = undefined, sender) {
      if (typeof msg === "string") {
        msg = { text: msg };
      }
      msg.sender = sender;
      this.msgList.push(msg);
      this.$nextTick(()=>{
        this.$refs.scrollRefs[this.$refs.scrollRefs.length-1].scrollIntoView();
      })
    },

    sendMsg(text) {
      this.createMsg(text, "user");
      this.$emit('disableSend'); // let user not able to send msg until responded
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
        /* =================test code====================*/
        // this.msgList = testdata;
        /* =================test code====================*/
        return;
      } else {
        // send http req for query in message saved
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

    switchRoute(to=null, from=null) {
      this.loadChat(to.params.cid);
      if (to.params.cid === "new") {
        this.generateSessionId()
        this.$router.replace({ name: "Chat", params: { cid: this.session_id }})
        return;
      }
      this.session_id = to.params.cid;
      this.initConnection();
    },
  },

  beforeRouteUpdate(to, from) {
    // console.log('route update! to ' +to.params.cid+' from '+from.params.cid)
    this.switchRoute(to, from);
  },

  beforeRouteEnter(to, from, next) {
    next((vm) => {
      /* =================test code====================*/
      // if (to.params.cid === "new") {
      //   vm.loadChat(to.params.cid);
      //   return;
      // }
      /* =================test code====================*/

      vm.checkLoginInfo(to, from);
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
