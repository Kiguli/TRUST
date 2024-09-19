<script setup>
import { onMounted, reactive, ref } from "vue";

import H2 from "@/Atoms/H2.vue";
import { dim } from "picocolors";
import Input from "@/Atoms/Input.vue";
import LabelledInput from "@/Molecules/LabelledInput.vue";
import Label from "@/Atoms/Label.vue";

defineProps({
    title: {
        type: String,
        required: true,
    },
    description: {
        type: String,
        default: "",
    },
    dimensions: {
        type: Number,
        required: true,
    },
});

const vectorData = ref([]);
const inputError = ref(false);

const updateMatrix = (i, event) => {
    inputError.value = false;

    const oldValue = event.target.value;

    if (!oldValue) {
        vectorData.value[i - 1] = [];
        return;
    }

    const parsed = oldValue
        .split(",")
        .map(Number)
        .filter((value) => value !== undefined);

    inputError.value = parsed.some((value) => isNaN(value));

    // If the input has more than two values, show an error
    if (parsed.length > 2) {
        inputError.value = true;
        return;
    }

    vectorData.value[i - 1] = parsed;
}
</script>

<template>
    <div class="py-1">
        <H2>{{ title }}</H2>
        <p class="text-sm text-gray-400">
            {{ description }}
        </p>
        <div
            v-for="i in dimensions"
            :key="i"
            class="mt-2 flex rounded-md shadow-sm">
            <Label
                :for="`x${i}`">
                x<sub class="mt-1">{{ i }}</sub>
            </Label>
            <Input
                :id="`x${i}`"
                @input="updateMatrix(i, $event)"
                aria-autocomplete="none"
                autocapitalize="off"
                autocomplete="off"
                :has-error="inputError"
                :placeholder="i === 1 ? 'e.g. 1, 2' : '...'"
                required
                type="text" />
        </div>
    </div>
</template>

<style scoped></style>
