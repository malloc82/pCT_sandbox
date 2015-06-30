(ns utils.core
  (:require [clojure.string :as str]
            [clojure.tools.cli :refer [parse-opts]]
            [clojure.pprint :refer [pprint]]
            [clojure.java.io :as io])
  (:gen-class))

(defn update-config [filename parameters]
  (let [para  (mapv #(str/split % #"=") (str/split parameters #",[\s]*"))
        para_count (count para)
        lines (str/split (slurp filename) #"[\n]")]
    (pprint para)
    (mapv (fn [lin]
            (loop [i (int 0)]
              (if (< i para_count)
                (if (.contains lin ((para i) 0))
                  (let [pattern (re-pattern (str/join ["\\s" ((para i) 0) "[\\s]*=[^;]+;"]))]
                    (str/replace lin pattern
                                 (str/join [" " ((para i) 0) " = " ((para i) 1) ";"])))
                  (recur (unchecked-inc i)))
                lin)))
          lines)))

;; (update-config "../include/pCT_config.h" "DEBUG_TEXT_ON=false,MODIFY_MLP=false")

(defn usage [summary]
  (pprint "summary:")
  (pprint summary))

(defn exit [status msg]
  (pprint msg)
  ;; (System/exit status)
  )

(def config-cli-options
  [[nil "--header HEADER_FILE" "header configuration"
    :parse-fn #(File. %)
    :validate [#(and (.exists %) (.isFile %)) "File must exists."]]
   ["-o" "--option PCT_OPTIONS" "pCT settings"]
   ["-y" "--dry" "Dry run"
    :id :dry]
   ["-h" "--help" "help"]])

(defn -main [& args]
  (let [mode (first args)]
    (println mode)
    (let [{:keys [options arguments errors summary]} (parse-opts (rest args) config-cli-options)]
      (cond
        (:help options) (exit 0 (usage summary))
        errors (exit 1 (pprint errors)))
      (pprint options)
      (.getPath (:header options))))
  (println "hello, world!"))

