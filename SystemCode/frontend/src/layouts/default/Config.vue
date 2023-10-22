<template>
  <VSonner position="top-center"></VSonner>
  <v-layout ref="app" class="rounded rounded-md">
    <v-main
      class="d-flex align-center justify-center"
      style="min-height: 600px"
    >
      <v-card class="my-10 px-5 py-3" title="Configure Telegram Notification">
        <v-card-text class="text-body-1 py-3 my-0"
          >Hi, {{ username }}!</v-card-text
        >
        <template v-if="verified">
          <v-alert type="success"
            >You've already subscribed to our Telegram reminder!<br />
            However you can set it up again by re-verifing with the code.<br />
            <v-btn
              :loading="loading"
              size="small"
              variant="outlined"
              density="compact"
              class="text-none"
              @click.stop="sendTestMsg"
              >Send my first day meal plan to me.</v-btn
            >
          </v-alert>
        </template>
        <template v-if="warning">
          <v-alert type="warning"
            >It seems that we haven't receive your code. Please try again.
          </v-alert>
        </template>
        <v-card-text class="text-body-1"
          >You can send below verification message to our Telegram bot,<br />
          to receive notification from there. Please click the button to<br />
          verify once you've sent the code message.
        </v-card-text>
        <v-card-text class="text-h6 pb-3">
          Send message: <span class="font-weight-bold">{{ veriCode }}</span
          ><br
        /></v-card-text>
        <v-card-text class="text-h6 pt-0"
          >to
          <a href="https://t.me/ai_health_monitor_bot"
            >@ai_health_monitor_bot</a
          >
          in Telegram.</v-card-text
        >
        <v-card-item>
          <v-btn variant="outlined" @click.stop="$router.back()"> Back </v-btn>
          <template v-slot:append
            ><v-btn
              :loading="loading"
              elevation="0"
              class="bg-primary text-none"
              @click.stop="sendVeri"
              >Already Sent, Verify</v-btn
            ></template
          >
        </v-card-item>
      </v-card>
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
    verified: true,
    warning: false,

    loading: false,

    username: undefined,
    veriCode: undefined,
    text: "",
  }),

  methods: {
    sendVeri() {
      this.loading = true;
      axios
        .post("http://" + hostname + "/api/curr-user-verify", {
          code: this.veriCode,
        })
        .then((res) => {
          if (res.data.failed) {
            alertToast("Verification failed!", "error");
            return;
          }
          if (res.data.verified) {
            this.verified = true;
            this.warning = false;
          } else {
            this.warning = true;
          }
        })
        .catch((res) => {
          alertToast("Verification failed!", "error");
        })
        .finally(() => {
          this.loading = false;
          this.generateCode();
        });
    },

    generateCode() {
      let chars = [
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "0",
        "a",
        "b",
        "c",
        "d",
        "e",
        "f",
      ];
      let code = "";
      for (let i = 0; i < 4; i++) {
        let randPos = Math.floor(Math.random() * (chars.length - 1));
        code += chars[randPos];
      }
      this.veriCode = code;
    },

    sendTestMsg() {
      this.loading = true;
      axios
        .post("http://" + hostname + "/api/send-tele-msg", { msg: "msg" })
        .then((res) => {
          this.loading = true;
          if (res.data.failed) {
            alertToast("Failed!", "error");
          }
          if (res.data.sent) {
            alertToast("Sent!", "success");
          }else{
            alertToast("You haven't had a meal plan yet!", "warning");
          }
        })
        .catch((res) => {
          alertToast("Failed!", "error");
        })
        .finally(() => {
          this.loading = false;
        });
    },
  },

  mounted() {
    axios
      .get("http://" + hostname + "/api/curr-user-notify")
      .then((res) => {
        this.username = res.data.username;
        if (res.data.configured) {
          this.verified = true;
        }
      })
      .catch((res) => {
        alertToast("Connection failed!", "error");
      })
      .finally(() => {
        this.generateCode();
      });
  },
};
</script>
